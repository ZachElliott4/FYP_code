from __future__ import annotations

import os, inspect
os.environ["TRANSFORMERS_NO_TF"] = "1"
if not hasattr(inspect, "formatargspec"):
    inspect.formatargspec = lambda *a, **k: ""

import os, struct, math, tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

import numpy as np, constriction, torch, torchaudio, soundfile as sf, whisper
from TTS.api import TTS as XTTS
from transformers import AutoTokenizer, AutoModelForCausalLM

XTTS_ID     = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
WHISPER_ID  = os.getenv("WHISPER_ID", "medium.en")
LM_ID       = os.getenv("LM_ID", "gpt2")
PMODE_TXT   = os.getenv("PACK_MODE_TXT", "lm").lower()
PMODE_AUD   = os.getenv("PACK_MODE_AUD", "lm").lower()
SEG_DUR     = float(os.getenv("SEG_DUR", "3"))
SEQ_LEN     = int(os.getenv("SEQ_LEN", "1023"))
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTN_IMPL   = os.getenv("LLM_ATTN_IMPL")
QUANT       = os.getenv("LLM_QUANT", "").lower()
BATCH       = max(1, int(os.getenv("LLM_BATCH", "1")))

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
_b2u = bytes_to_unicode()
bytes2text = lambda bs: ''.join(_b2u[b] for b in bs)
MU_MAX = 255.0

def mulaw_encode(wav: np.ndarray) -> bytes:
    """8‑bit µ‑law, return raw bytes"""
    mu = 255
    mag = np.log1p(mu * np.abs(wav)) / np.log1p(mu)
    signal = np.sign(wav) * mag
    enc = ((signal + 1) / 2 * mu + 0.5).astype(np.uint8)
    return enc.tobytes()

def mulaw_decode(bs: bytes) -> np.ndarray:
    mu = 255
    enc = np.frombuffer(bs, dtype=np.uint8).astype(np.float32)
    signal = 2 * (enc / mu) - 1
    mag = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    wav = np.sign(signal) * mag
    return wav

@lru_cache(maxsize=1)
def _load_whisper():
    return whisper.load_model(WHISPER_ID, device=DEVICE)

@lru_cache(maxsize=1)
def _load_xtts():
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])

    _orig_load = torch.load
    torch.load = lambda p, *a, **k: _orig_load(p, map_location="cpu", weights_only=False)

    tts = XTTS(model_name=XTTS_ID, gpu=(DEVICE.type == "cuda"))

    torch.load = _orig_load
    return tts

@lru_cache(maxsize=1)
def _load_lm():
    tok = AutoTokenizer.from_pretrained(LM_ID)
    kw: Dict[str, Any] = dict(
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        device_map="auto" if DEVICE.type == "cuda" else None,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    if ATTN_IMPL:
        kw["attn_implementation"] = ATTN_IMPL
    if QUANT == "8bit": kw["load_in_8bit"] = True
    if QUANT == "4bit": kw["load_in_4bit"] = True
    lm = AutoModelForCausalLM.from_pretrained(LM_ID, **kw).eval()
    return tok, lm

def encode(wav: np.ndarray, sr: int) -> Dict[str, Any]:
    asr = _load_whisper()
    wav_t = torch.from_numpy(wav).to(DEVICE)
    if sr != 16_000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16_000)
    text = asr.transcribe(audio=wav_t, fp16=(DEVICE.type == "cuda"))["text"].strip()

    seg_len = int(SEG_DUR * sr)
    seg = wav[:seg_len]
    if len(seg) < seg_len:
        seg = np.pad(seg, (0, seg_len - len(seg)))

    seg_mono = seg if seg.ndim == 1 else seg.mean(0)
    seg8   = torchaudio.functional.resample(torch.from_numpy(seg_mono), sr, 8_000).numpy()
    seg_bytes = mulaw_encode(seg8)

    tok, _ = _load_lm()
    tokens = tok.encode(text, add_special_tokens=False)

    return {
        "tokens": tokens,
        "seg_bytes": seg_bytes,
        "sr_seg": 8_000,
        "orig_sr": sr,
        "text_len": len(tokens),
        "seg_len": len(seg_bytes),
    }

def _encode_stream(symbols: List[int], probs_fn) -> bytes:
    enc = constriction.stream.queue.RangeEncoder()
    for s in symbols:
        probs = probs_fn()
        enc.encode(s, constriction.stream.model.Categorical(probs.astype(np.float32), perfect=False))
    return enc.get_compressed()

def pack(info: Dict[str, Any]) -> bytes:
    tok, lm = _load_lm(); EOS = tok.eos_token_id
    txt_enc = constriction.stream.queue.RangeEncoder()
    if PMODE_TXT == "uniform":
        vocab = lm.config.vocab_size
        uni = np.full(vocab, 1 / vocab, np.float32)
        for t in info["tokens"]:
            txt_enc.encode(t, constriction.stream.model.Categorical(uni, perfect=False))
    else:
        blocks = [info["tokens"][i:i+SEQ_LEN] for i in range(0, info["text_len"], SEQ_LEN)]
        ctx_prev = [EOS]
        with torch.inference_mode():
            for blk in blocks:
                inp = torch.tensor([ctx_prev + blk], device=DEVICE)
                probs_blk = torch.softmax(lm(inp).logits[0, -len(blk):].float(), -1).cpu().numpy()
                for pos, tid in enumerate(blk):
                    txt_enc.encode(tid, constriction.stream.model.Categorical(probs_blk[pos].astype(np.float32), perfect=False))
                ctx_prev = blk[-1:]
    txt_arr = txt_enc.get_compressed()
    txt_stream = txt_arr.tobytes() if isinstance(txt_arr, np.ndarray) else bytes(txt_arr)

    aud_enc = constriction.stream.queue.RangeEncoder()
    if PMODE_AUD == "uniform":
        uni256 = np.full(256, 1 / 256, np.float32)
        for b in info["seg_bytes"]:
            aud_enc.encode(b, constriction.stream.model.Categorical(uni256, perfect=False))
    else:
        aud_tokens = tok.encode(bytes2text(info["seg_bytes"]), add_special_tokens=False)
        blk_aud = [aud_tokens[i:i + SEQ_LEN] for i in range(0, len(aud_tokens), SEQ_LEN)]
        ctx_prev = [EOS]
        with torch.inference_mode():
            for blk in blk_aud:
                inp = torch.tensor([ctx_prev + blk], device=DEVICE)
                probs_a = torch.softmax(lm(inp).logits[0, -len(blk):].float(), -1).cpu().numpy()
                for pos, tid in enumerate(blk):
                    aud_enc.encode(tid, constriction.stream.model.Categorical(
                        probs_a[pos].astype(np.float32), perfect=False))
                ctx_prev = blk[-1:]
    aud_arr = aud_enc.get_compressed()
    aud_stream = aud_arr.tobytes() if isinstance(aud_arr, np.ndarray) else bytes(aud_arr)

    header = struct.pack("<II", len(txt_stream), len(aud_stream))
    payload = header + txt_stream + aud_stream
    return payload

def decode(info: Dict[str, Any], sr: int) -> np.ndarray:
    tok, _ = _load_lm()
    text = tok.decode(info["tokens"])

    seg_pcm = mulaw_decode(info["seg_bytes"])
    seg_pcm = torchaudio.functional.resample(torch.from_numpy(seg_pcm), info["sr_seg"], 24_000).numpy()
    seg_path = Path(tempfile.mktemp(suffix="_ref.wav"))
    sf.write(seg_path, seg_pcm, 24_000)

    xtts = _load_xtts()
    speech = xtts.tts(text=text, speaker_wav=str(seg_path), language="en")
    torchaudio.save(seg_path.with_suffix(".tmp.wav"), torch.tensor(speech).unsqueeze(0), 24_000)
    os.remove(seg_path)
    if isinstance(speech, list):
        parts = [np.atleast_1d(np.asarray(p, dtype=np.float32)) for p in speech]
        speech = np.concatenate(parts, axis=-1)
    else:
        speech = speech.astype(np.float32)
    return speech

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("XTTS wrapper smoke test")
    parser.add_argument("wav")
    parser.add_argument("--outdir")
    a = parser.parse_args()

    wav, sr0 = sf.read(a.wav, dtype="float32", always_2d=False)
    info = encode(wav, sr0)
    payload = pack(info)
    recon = decode(info, sr0)

    outdir = Path(a.outdir) if a.outdir else Path(tempfile.mkdtemp(prefix="xttswrap_"))
    outdir.mkdir(parents=True, exist_ok=True)
    sf.write(outdir / "recon.wav", recon, 24_000)
    (outdir / "payload.bin").write_bytes(payload)

    kbps = 8 * len(payload) / (len(wav) / sr0) / 1000
    print(f"✓ {len(payload)} bytes ({kbps:.2f} kbps) → {outdir}")
