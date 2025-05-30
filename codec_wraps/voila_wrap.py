from __future__ import annotations
import os, numpy as np, torch, torchaudio, constriction
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, EncodecModel
from functools import lru_cache

DEVICE   = torch.device(os.getenv("VOILA_DEVICE", "cpu"))
BW       = float(os.getenv("VOILA_BW", "1.1"))
LM_ID    = os.getenv("VOILA_LM",    "maitrix-org/Voila-base")
CODEC_ID = os.getenv("VOILA_CODEC", "maitrix-org/Voila-Tokenizer")
TARGET_SR = 16_000

@lru_cache(maxsize=1)
def _load_lm():
    tok = AutoTokenizer.from_pretrained(LM_ID, trust_remote_code=True)
    lm  = AutoModelForCausalLM.from_pretrained(
            LM_ID,
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
          ).to(DEVICE).eval()
    return tok, lm

@lru_cache(maxsize=1)
def _load_codec():
    codec = EncodecModel.from_pretrained(CODEC_ID).to(DEVICE)
    proc  = AutoProcessor.from_pretrained(CODEC_ID)
    return codec, proc.sampling_rate

def encode(wav: np.ndarray, sr: int) -> dict:
    codec, _ = _load_codec()
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr, TARGET_SR).numpy()
    wav_t = torch.from_numpy(wav).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    codes2d = codec.encode(wav_t, bandwidth=BW).audio_codes[0, 0]
    return {"codes": codes2d.cpu().long(), "n_q": codes2d.shape[0], "length": len(wav)}

def decode(pack: dict, sr: int) -> np.ndarray:
    codec, _ = _load_codec()
    codes2d  = pack["codes"].to(DEVICE)
    wav = codec.decode(codes2d[None, None], [None])[0, 0]
    wav = wav[: pack["length"]].cpu().clamp_(-1.0, 1.0).numpy().astype(np.float32)
    return wav

def pack(bitstream: dict) -> bytes:
    """LM range-encode the flattened token list and return raw bytes."""
    tok, lm   = _load_lm()
    flat      = bitstream["codes"].flatten().tolist()
    ids       = [tok.eos_token_id] + flat + [tok.eos_token_id]
    enc       = constriction.stream.queue.RangeEncoder()
    with torch.inference_mode():
        for i in range(1, len(ids)):
            ctx   = torch.tensor(ids[:i], device=DEVICE).unsqueeze(0)
            probs = torch.softmax(lm(ctx).logits[0, -1].float(), -1).cpu().numpy()
            enc.encode(ids[i],
                       constriction.stream.model.Categorical(
                           probs.astype(np.float32), perfect=False))
    return enc.get_compressed()
