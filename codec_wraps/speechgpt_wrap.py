#!/usr/bin/env python3
"""
SpeechGPT codec
"""
from __future__ import annotations

import json, os, urllib.request, tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import constriction, faiss, joblib, numpy as np, soundfile as sf, torch, torchaudio
from fairseq import checkpoint_utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from fairseq.data.dictionary import Dictionary
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.serialization.add_safe_globals([Dictionary])

# user‑configurable env vars
LM_ID      = os.getenv("SGPT_LM", "fnlp/SpeechGPT-7B-cm")
DEVICE     = torch.device(os.getenv("SGPT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
STEP       = int(os.getenv("SGPT_FRAME_STEP", "1"))        # frame skip (1=20 ms)
PACK_MODE  = os.getenv("SGPT_PACK", "lm").lower()           # "lm" or "uniform"

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"; ASSET_DIR.mkdir(exist_ok=True)
ASSETS = {
    "mhubert":  (ASSET_DIR / "mhubert_base.pt", "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt"),
    "kmeans":   (ASSET_DIR / "mhubert_L11_km1000.bin", "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"),
    "voc_ckpt": (ASSET_DIR / "vocoder.pt", "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000"),
    "voc_cfg":  (ASSET_DIR / "vocoder_config.json", "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json"),
}

def _download_assets():
    for name, (path, url) in ASSETS.items():
        if not path.exists():
            print(f"[SpeechGPT] downloading {name} …")
            urllib.request.urlretrieve(url, path)

@lru_cache(maxsize=1)
def _load_feature_extractor():
    _download_assets()
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(ASSETS["mhubert"][0])])
    hubert = models[0].to(DEVICE).eval()
    kmeans = joblib.load(ASSETS["kmeans"][0])
    index = faiss.IndexFlatL2(kmeans.cluster_centers_.shape[1])
    index.add(np.asarray(kmeans.cluster_centers_, dtype="float32"))
    return hubert, index

@lru_cache(maxsize=1)
def _load_lm():
    tok = AutoTokenizer.from_pretrained(LM_ID, trust_remote_code=True)
    lm  = AutoModelForCausalLM.from_pretrained(
            LM_ID,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
            device_map="auto" if DEVICE.type == "cuda" else None,
            trust_remote_code=True,
            use_safetensors=False,
    ).eval()
    return tok, lm

@lru_cache(maxsize=1)
def _load_vocoder():
    _download_assets()
    with open(ASSETS["voc_cfg"][0]) as f: cfg = json.load(f)
    _orig = torch.load; torch.load = lambda p,*a,**k: _orig(p, map_location="cpu")
    voc = CodeHiFiGANVocoder(ASSETS["voc_ckpt"][0], cfg).to(DEVICE).eval()
    torch.load = _orig
    return voc

def encode(wav: np.ndarray, sr: int) -> Dict[str, Any]:
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, 16000).numpy()
    wav_t = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)

    hubert, index = _load_feature_extractor()
    with torch.no_grad():
        feats = hubert(wav_t, mask=False, features_only=True)
        feats = (feats["layer_results"][11][0] if "layer_results" in feats else feats["x"])[0]
        feats = feats[::STEP].cpu().numpy()
    _, I = index.search(feats, 1)
    return {"units": I.squeeze(1).tolist(), "length": len(wav)}

def pack(info: Dict[str, Any]) -> bytes:
    tok, lm = _load_lm(); units = info["units"]
    ids = [tok.convert_tokens_to_ids("<sosp>")] + tok.convert_tokens_to_ids([f"<{u}>" for u in units]) + [tok.convert_tokens_to_ids("<eosp>")]
    enc = constriction.stream.queue.RangeEncoder()

    if PACK_MODE == "uniform":
        vocab = lm.config.vocab_size
        uni   = np.full(vocab, 1/vocab, dtype=np.float32)
        for tid in ids[1:]: enc.encode(tid, constriction.stream.model.Categorical(uni, perfect=False))
        return enc.get_compressed()

    with torch.inference_mode():
        out = lm(torch.tensor([[ids[0]]], device=DEVICE), use_cache=True)
        past = out.past_key_values
        enc.encode(ids[1], constriction.stream.model.Categorical(torch.softmax(out.logits[0,-1].float(),-1).cpu().numpy().astype(np.float32), perfect=False))
        prev = ids[1]
        for tid in ids[2:]:
            out = lm(torch.tensor([[prev]], device=DEVICE), past_key_values=past, use_cache=True)
            past = out.past_key_values
            enc.encode(tid, constriction.stream.model.Categorical(torch.softmax(out.logits[0,-1].float(),-1).cpu().numpy().astype(np.float32), perfect=False))
            prev = tid
    return enc.get_compressed()

def decode(info: Dict[str, Any], sr: int) -> np.ndarray:
    voc = _load_vocoder(); units = info["units"]
    with torch.inference_mode():
        wav = voc({"code": torch.tensor(units, device=DEVICE).unsqueeze(0)}, dur_prediction=True).cpu()
    return wav.squeeze(0).numpy().astype(np.float32)

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="SpeechGPT wrapper smoke‑test")
    ap.add_argument("wav", type=Path, help="any audio file")
    ap.add_argument("--outdir", type=Path, help="where to dump recon.wav + bitstream")
    a = ap.parse_args()

    audio, sr0 = sf.read(a.wav, dtype="float32", always_2d=False)

    info    = encode(audio, sr0)
    payload = pack(info)
    recon   = decode(info, sr0)

    outdir = a.outdir or Path(tempfile.mkdtemp(prefix="speechgpt_"))
    outdir.mkdir(parents=True, exist_ok=True)
    sf.write(outdir / "recon.wav", recon, sr0)
    (outdir / "bitstream.bin").write_bytes(payload)

    kbps = 8*len(payload)/(len(audio)/sr0)/1000
    print(f"✓ Round‑trip OK – {len(payload)} bytes ({kbps:.2f} kb/s). Files in {outdir}")
