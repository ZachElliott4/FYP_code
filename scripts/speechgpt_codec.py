import argparse, pathlib, json, os, urllib.request, sys

import torch, torchaudio, faiss, joblib, numpy as np, constriction
from transformers import AutoTokenizer, AutoModelForCausalLM
from fairseq import checkpoint_utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from fairseq.data.dictionary import Dictionary

torch.serialization.add_safe_globals([Dictionary])

ASSET_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets"
ASSET_DIR.mkdir(exist_ok=True)

ASSETS = {
    "mhubert_pt": (
        ASSET_DIR / "mhubert_base.pt",
        "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt",
    ),
    "km_bin": (
        ASSET_DIR / "mhubert_L11_km1000.bin",
        "https://dl.fbaipublicfiles.com/hubert/"
        "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
    ),
    "vocoder_ckpt": (
        ASSET_DIR / "vocoder.pt",
        "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/"
        "code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000",
    ),
    "vocoder_cfg": (
        ASSET_DIR / "vocoder_config.json",
        "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/"
        "code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json",
    ),
}

LM_ID = "fnlp/SpeechGPT-7B-cm"

def download_if_missing():
    for name, (path, url) in ASSETS.items():
        if not path.exists():
            print(f"Downloading {name} …")
            urllib.request.urlretrieve(url, path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--outdir", type=pathlib.Path, default="outputs/speechgpt")
    p.add_argument("--step", type=int, default=1, help="frame-skip for tokens")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    download_if_missing()

    # 1) Load mHuBERT + k-means
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(ASSETS["mhubert_pt"][0])]
    )
    hubert = models[0].to(args.device).eval()

    kmeans = joblib.load(ASSETS["km_bin"][0])
    centroids = np.asarray(kmeans.cluster_centers_, dtype="float32")
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)

    # 2) Audio → tokens
    wav, sr = torchaudio.load(str(args.input))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0).unsqueeze(0)

    with torch.no_grad():
        out = hubert(wav.to(args.device), mask=False, features_only=True)
        feats = out["layer_results"][11][0] if "layer_results" in out else out["x"]
        feats = feats[0][:: args.step].cpu().numpy()

    _, I = index.search(feats, 1)
    unit_ids = I.squeeze(1).tolist()
    print("Tokenised:", len(unit_ids), "units")

    # 3) Load SpeechGPT + tokenizer
    tok = AutoTokenizer.from_pretrained(LM_ID, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        LM_ID,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
        device_map="auto" if "cuda" in args.device else None,
        trust_remote_code=True,
    ).eval()

    sosp = tok.convert_tokens_to_ids("<sosp>")
    eosp = tok.convert_tokens_to_ids("<eosp>")
    str_tokens = [f"<{i}>" for i in unit_ids]
    input_ids = [sosp] + tok.convert_tokens_to_ids(str_tokens) + [eosp]

    # 4) Arithmetic-encode with LM
    enc = constriction.stream.queue.RangeEncoder()
    for i in range(1, len(input_ids)):
        ctx = torch.tensor([input_ids[:i]], device=args.device)
        logits = lm(ctx).logits[0, -1]
        probs = torch.softmax(logits.float(), -1).cpu().numpy()
        enc.encode(input_ids[i], constriction.stream.model.Categorical(probs, False))

    bitstream = enc.get_compressed()
    bpt_lm = 8 * len(bitstream) / len(unit_ids)
    print(f"LM coding: {bpt_lm:.2f} bits/token")

    # 5) Uniform baseline
    enc_u = constriction.stream.queue.RangeEncoder()
    vocab = lm.config.vocab_size
    uniform = np.ones(vocab, dtype=np.float32) / vocab
    for tid in input_ids[1:]:
        enc_u.encode(tid, constriction.stream.model.Categorical(uniform, False))
    bitstream_u = enc_u.get_compressed()
    bpt_u = 8 * len(bitstream_u) / len(unit_ids)
    print(f"Uniform:   {bpt_u:.2f} bits/token  "
          f" -> gain {bpt_u - bpt_lm:.2f} ({(bpt_u-bpt_lm)/bpt_u:.1%})")

    # 6) Decode & vocoder
    dec, decoded = constriction.stream.queue.RangeDecoder(bitstream), [sosp]
    for _ in range(len(unit_ids) + 1):
        ctx = torch.tensor([decoded], device=args.device)
        probs = torch.softmax(lm(ctx).logits[0, -1].float(), -1).cpu().numpy()
        decoded.append(
            dec.decode(constriction.stream.model.Categorical(probs.astype(np.float32)))
        )
    assert decoded == input_ids, "round-trip mismatch"

    voc_cfg_path, voc_ckpt_path = ASSETS["vocoder_cfg"][0], ASSETS["vocoder_ckpt"][0]
    with open(voc_cfg_path) as f:
        voc_cfg = json.load(f)
    _orig_load = torch.load
    torch.load = lambda p, *a, **k: _orig_load(p, map_location="cpu")
    vocoder = CodeHiFiGANVocoder(voc_ckpt_path, voc_cfg).eval()
    torch.load = _orig_load

    unit_tensor = torch.tensor(unit_ids, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        wav_hat = vocoder({"code": unit_tensor}, dur_prediction=True).squeeze(0).cpu()
    out_wav = args.outdir / "recon.wav"
    torchaudio.save(str(out_wav), wav_hat, 16000)

    # 7) Save stats
    stats = {
        "n_units": len(unit_ids),
        "bpt_lm": bpt_lm,
        "bpt_uniform": bpt_u,
    }
    (args.outdir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("Done; files in", args.outdir)


if __name__ == "__main__":
    main()
