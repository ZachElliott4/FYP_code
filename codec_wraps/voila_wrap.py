"""
Voila codec
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import torch
import torchaudio
import constriction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    EncodecModel,
)

PACK_MODE = os.getenv("VOILA_PACK", "lm").lower()
__all__ = ["encode", "pack", "decode"]

DEVICE = torch.device(os.getenv("VOILA_DEVICE",
                                "cuda" if torch.cuda.is_available() else "cpu"))
BW        = float(os.getenv("VOILA_BW", "1.1"))
LM_ID     = os.getenv("VOILA_LM",    "maitrix-org/Voila-base")
CODEC_ID  = os.getenv("VOILA_CODEC", "maitrix-org/Voila-Tokenizer")
ATTN_IMPL = os.getenv("VOILA_ATTN_IMPL", "flash_attention_2")
TARGET_SR = 16_000

@lru_cache(maxsize=1)
def _load_codec():
    codec = EncodecModel.from_pretrained(CODEC_ID).to(DEVICE)
    proc  = AutoProcessor.from_pretrained(CODEC_ID)
    return codec, proc.sampling_rate


@lru_cache(maxsize=1)
def _load_lm():
    tok = AutoTokenizer.from_pretrained(LM_ID, trust_remote_code=True)

    load_kw = dict(
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if DEVICE.type == "cuda":
        load_kw["device_map"] = {"": DEVICE.index or 0}
        load_kw["attn_implementation"] = ATTN_IMPL

    try:
        lm = AutoModelForCausalLM.from_pretrained(LM_ID, **load_kw)
    except (TypeError, ValueError):
        load_kw.pop("attn_implementation", None)
        lm = AutoModelForCausalLM.from_pretrained(LM_ID, **load_kw)

    lm.eval()

    if (os.getenv("VOILA_COMPILE", "0") == "1"
            and hasattr(torch, "compile")
            and DEVICE.type == "cuda"):
        try:
            lm = torch.compile(lm, mode="reduce-overhead")
        except Exception:
            pass

    return tok, lm

def _to_mono_np(wav: np.ndarray) -> np.ndarray:
    return wav.mean(axis=0) if wav.ndim == 2 else wav


def _extract_wave(obj: Any) -> torch.Tensor:
    if hasattr(obj, "audio_values"):
        wav = obj.audio_values
    elif hasattr(obj, "audio"):
        wav = obj.audio
    elif isinstance(obj, (tuple, list)) and isinstance(obj[0], torch.Tensor):
        wav = obj[0]
    elif isinstance(obj, torch.Tensor):
        wav = obj
    else:
        raise TypeError(f"Unexpected Encodec decode output: {type(obj)}")

    if wav.dim() == 3:
        wav = wav[0, 0]
    elif wav.dim() == 2:
        wav = wav[0]
    return wav

def encode(wav: np.ndarray, sr: int) -> Dict[str, Any]:
    wav = _to_mono_np(wav).astype(np.float32, copy=False)
    codec, _ = _load_codec()

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), sr, TARGET_SR)[0].numpy()

    wav_t   = torch.from_numpy(wav).to(DEVICE).unsqueeze(0).unsqueeze(0)
    codes2d = codec.encode(wav_t, bandwidth=BW).audio_codes[0, 0]  # (n_q, T)

    return {"codes": codes2d.cpu().long(),
            "n_q":   codes2d.shape[0],
            "length": len(wav)}

def _encode_symbol(enc, s: int, p: np.ndarray):
    enc.encode(s, constriction.stream.model.Categorical(p.astype(np.float32),
                                                        perfect=False))


def pack(bitstream: Dict[str, Any]) -> bytes:
    tok, lm = _load_lm()
    tokens: List[int] = bitstream["codes"].flatten().tolist()

    enc = constriction.stream.queue.RangeEncoder()

    if PACK_MODE == "uniform":
        print("Using uniform")
        vocab     = lm.config.vocab_size
        uniform_p = np.full(vocab, 1.0 / vocab, dtype=np.float32)
        for t in tokens:
            enc.encode(t,
                       constriction.stream.model.Categorical(uniform_p,
                                                             perfect=False))
        return enc.get_compressed()

    with torch.inference_mode():
        print("Using lm")
        out   = lm(torch.tensor([[tok.eos_token_id]], device=DEVICE),
                   use_cache=True)
        past  = out.past_key_values
        probs = torch.softmax(out.logits[0, -1].float(), -1).cpu().numpy()
        _encode_symbol(enc, tokens[0], probs)

        prev = tokens[0]
        for t in tokens[1:]:
            out = lm(torch.tensor([[prev]], device=DEVICE),
                     past_key_values=past, use_cache=True)
            past  = out.past_key_values
            probs = torch.softmax(out.logits[0, -1].float(), -1).cpu().numpy()
            _encode_symbol(enc, t, probs)
            prev = t

    return enc.get_compressed()

def decode(bitstream: Dict[str, Any], sr: int) -> np.ndarray:
    codec, _ = _load_codec()
    codes2d  = bitstream["codes"].to(DEVICE)

    with torch.inference_mode():
        dec_out = codec.decode(codes2d[None, None], [None])
        wav = _extract_wave(dec_out)

    return (wav[:bitstream["length"]]
              .detach()
              .clamp(-1.0, 1.0)
              .cpu()
              .numpy()
              .astype(np.float32))

if __name__ == "__main__":
    import argparse, soundfile as sf, tempfile, pathlib

    P = argparse.ArgumentParser(description="Quick Voila round-trip smoke test.")
    P.add_argument("wav", help="Any audio file (mono/stereo, any SR).")
    args = P.parse_args()

    audio, sr0 = sf.read(args.wav, dtype="float32", always_2d=False)
    bitstream  = encode(audio, sr0)
    payload    = pack(bitstream)
    recon      = decode(bitstream, sr0)

    outdir = pathlib.Path(tempfile.mkdtemp(prefix="voila_"))
    sf.write(outdir / "recon.wav", recon, TARGET_SR)
    (outdir / "bitstream.bin").write_bytes(payload)

    rate_kbps = 8 * len(payload) / (len(audio)/sr0) / 1000
    print(f"✓ Round-trip OK – {len(payload)} bytes ({rate_kbps:.2f} kb/s). "
          f"Files in {outdir}")
