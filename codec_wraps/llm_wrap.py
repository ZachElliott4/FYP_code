#!/usr/bin/env python3
"""
LLM compression
"""
from __future__ import annotations

import os, tempfile, time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import constriction, numpy as np, torch, soundfile as sf
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

LLM_ID   = os.getenv("LLM_ID", "gpt2")
PACK_MODE = os.getenv("LLM_PACK", "lm").lower()
ATTN_IMPL = os.getenv("LLM_ATTN_IMPL")
QUANT     = os.getenv("LLM_QUANT", "").lower()
BATCH     = max(1, int(os.getenv("LLM_BATCH", "1")))
SEQ_LEN   = int(os.getenv("SEQ_LEN", "1023"))
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_b2u = bytes_to_unicode(); _u2b = {v: k for k, v in _b2u.items()}
def bytes2text(bs: bytes) -> str:   return ''.join(_b2u[b] for b in bs)
def text2bytes(txt: str) -> bytes:  return bytes(_u2b[ch] for ch in txt)

@lru_cache(maxsize=1)
def _load_lm():
    tok = AutoTokenizer.from_pretrained(LLM_ID)

    kw = dict(
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        device_map="auto" if DEVICE.type == "cuda" else None,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    if ATTN_IMPL:
        kw["attn_implementation"] = ATTN_IMPL
    if QUANT == "8bit":
        kw["load_in_8bit"] = True
    elif QUANT == "4bit":
        kw["load_in_4bit"] = True

    lm = AutoModelForCausalLM.from_pretrained(LLM_ID, **kw).eval()
    return tok, lm

def encode(wav: np.ndarray, sr: int) -> Dict[str, any]:
    raw_bytes = wav.astype(np.float32, copy=False).tobytes()
    txt = bytes2text(raw_bytes)
    tok, _ = _load_lm()
    ids = tok.encode(txt, add_special_tokens=False)
    return {"codes": ids, "n_samples": len(wav), "sr": sr}

def _encode_uniform(enc, ids: List[int], vocab: int):
    uni = np.full(vocab, 1 / vocab, dtype=np.float32)
    for tid in ids:
        enc.encode(tid, constriction.stream.model.Categorical(uni, perfect=False))

def pack(info: Dict[str, any]) -> bytes:
    ids = info["codes"]
    tok, lm = _load_lm(); EOS = tok.eos_token_id
    blocks = [ids[i:i + SEQ_LEN] for i in range(0, len(ids), SEQ_LEN)]
    enc = constriction.stream.queue.RangeEncoder()

    if PACK_MODE == "uniform":
        _encode_uniform(enc, ids, lm.config.vocab_size)
        return enc.get_compressed()

    ctx_prev = [EOS]
    with torch.inference_mode():
        for i in range(0, len(blocks), BATCH):
            for blk in blocks[i:i + BATCH]:
                inp = torch.tensor([ctx_prev + blk], device=DEVICE)
                logits = lm(inp).logits[0, -len(blk):]

                probs_block = torch.softmax(logits.float(), -1).cpu().numpy()
                for pos, tid in enumerate(blk):
                    enc.encode(
                        tid,
                        constriction.stream.model.Categorical(
                            probs_block[pos].astype(np.float32),
                            perfect=False,
                        ),
                    )
                ctx_prev = blk[-1:]
    return enc.get_compressed()

def decode(info: Dict[str, any], sr: int) -> np.ndarray:
    # reconstruct waveform directly from tokens (skips arithmetic decoding)
    tok, _ = _load_lm()
    wav_bytes = text2bytes(tok.decode(info["codes"]))
    return np.frombuffer(wav_bytes, dtype=np.float32, count=info["n_samples"])

if __name__ == "__main__":
    import argparse, tempfile
    P = argparse.ArgumentParser(description="Byte-LLM wrapper smoke-test")
    P.add_argument("wav")
    P.add_argument("--outdir", type=Path)
    a = P.parse_args()

    wav, sr = sf.read(a.wav, dtype="float32", always_2d=False)
    t0 = time.time()
    info = encode(wav, sr)
    payload = pack(info)
    recon = decode(info, sr)
    dt = time.time() - t0

    outdir = a.outdir or Path(tempfile.mkdtemp(prefix="llmbytes_"))
    outdir.mkdir(parents=True, exist_ok=True)
    sf.write(outdir / "recon.wav", recon, sr)
    (outdir / "bitstream.bin").write_bytes(payload)

    br = 8 * len(payload) / (len(wav) / sr) / 1000
    print(f"✓ {len(payload)} bytes | {br:.2f} kbps | {dt:.1f} s  → {outdir}")
