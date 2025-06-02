from __future__ import annotations

import os, inspect, struct, tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any

import numpy as np, torch, constriction, torchaudio, whisper, soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM

WHISPER_ID = os.getenv("WHISPER_ID", "medium.en")
LM_ID      = os.getenv("LM_ID", "gpt2")
PACK_MODE  = os.getenv("PACK_MODE", "lm").lower()
SEQ_LEN    = int(os.getenv("SEQ_LEN", "1023"))
BATCH      = max(1, int(os.getenv("LLM_BATCH", "1")))
ATTN_IMPL  = os.getenv("LLM_ATTN_IMPL")
QUANT      = os.getenv("LLM_QUANT", "").lower()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BARK_DEVICE = torch.device(os.getenv("BARK_DEVICE", DEVICE.type))

@lru_cache(maxsize=1)
def _load_whisper():
    return whisper.load_model(WHISPER_ID, device=DEVICE)

@lru_cache(maxsize=1)
def _load_lm():
    tok = AutoTokenizer.from_pretrained(LM_ID)

    kw: Dict[str, Any] = dict(
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    if DEVICE.type == "cuda":
        kw["device_map"] = {"": DEVICE.index or 0}
        if ATTN_IMPL:
            kw["attn_implementation"] = ATTN_IMPL
    if QUANT == "8bit":
        kw["load_in_8bit"] = True
    elif QUANT == "4bit":
        kw["load_in_4bit"] = True

    lm = AutoModelForCausalLM.from_pretrained(LM_ID, **kw).eval()
    return tok, lm

@lru_cache(maxsize=1)
def _load_bark():
    from bark.generation import preload_models, generate_audio

    _orig_load = torch.load
    torch.load = lambda f, *a, **k: _orig_load(f, *a, **{**k, "weights_only": False})

    preload_models()

    torch.load = _orig_load
    return generate_audio

from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
_b2u = bytes_to_unicode()
bytes2text = lambda bs: ''.join(_b2u[b] for b in bs)

def encode(wav: np.ndarray, sr: int) -> Dict[str, Any]:
    asr = _load_whisper()
    wav_t = torch.from_numpy(wav)
    if wav.ndim == 2:
        wav_t = wav_t.mean(0)
    if sr != 16_000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16_000)
    wav_t = wav_t.to(DEVICE)
    text = asr.transcribe(audio=wav_t, fp16=(DEVICE.type == "cuda"))["text"].strip()

    tok, _ = _load_lm()
    tokens = tok.encode(text, add_special_tokens=False)

    return {
        "tokens": tokens,
        "length": len(tokens),
    }

def pack(info: Dict[str, Any]) -> bytes:
    tok, lm = _load_lm()
    token_ids: List[int] = info["tokens"]
    enc = constriction.stream.queue.RangeEncoder()

    if PACK_MODE == "uniform":
        vocab = lm.config.vocab_size
        uni   = np.full(vocab, 1 / vocab, dtype=np.float32)
        for tid in token_ids:
            enc.encode(tid, constriction.stream.model.Categorical(uni, perfect=False))
        return enc.get_compressed()

    EOS = tok.eos_token_id
    blocks = [token_ids[i:i + SEQ_LEN] for i in range(0, len(token_ids), SEQ_LEN)]
    ctx_prev = [EOS]
    with torch.inference_mode():
        for blk in blocks:
            inp = torch.tensor([ctx_prev + blk], device=DEVICE)
            logits = lm(inp).logits[0, -len(blk):]
            probs_blk = torch.softmax(logits.float(), -1).cpu().numpy()
            for pos, tid in enumerate(blk):
                enc.encode(tid, constriction.stream.model.Categorical(
                    probs_blk[pos].astype(np.float32), perfect=False))
            ctx_prev = blk[-1:]
    return enc.get_compressed()

def decode(info: Dict[str, Any], sr: int | None = None) -> np.ndarray:
    tok, _ = _load_lm()
    text = tok.decode(info["tokens"])

    generate_audio = _load_bark()
    with torch.no_grad():
        wav_24k = generate_audio(text)

    return wav_24k.astype(np.float32)

if __name__ == "__main__":
    import argparse, time

    P = argparse.ArgumentParser("Bark wrapper smoke-test")
    P.add_argument("wav", help="any audio file")
    P.add_argument("--outdir", type=Path)
    a = P.parse_args()

    audio, sr0 = sf.read(a.wav, dtype="float32", always_2d=False)

    t0 = time.time()
    info    = encode(audio, sr0)
    payload = pack(info)
    recon   = decode(info, sr0)
    dt = time.time() - t0

    outdir = a.outdir or Path(tempfile.mkdtemp(prefix="barkwrap_"))
    outdir.mkdir(parents=True, exist_ok=True)
    sf.write(outdir / "recon.wav", recon, 24_000)
    (outdir / "payload.bin").write_bytes(payload)

    kbps = 8 * len(payload) / (len(audio) / sr0) / 1000
    print(f"✓ {len(payload)} bytes ({kbps:.2f} kbps) | {dt:.1f} s → {outdir}")
