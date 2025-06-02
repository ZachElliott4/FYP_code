from __future__ import annotations

import os, struct, math, tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any

import numpy as np, constriction, torch, soundfile as sf, torchaudio, whisper
from speechbrain.inference import EncoderClassifier
from transformers import (
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,
    AutoTokenizer, AutoModelForCausalLM,
)
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

WHISPER_ID   = os.getenv("WHISPER_ID",   "medium.en")
SPK_ID       = os.getenv("SPK_MODEL_ID", "speechbrain/spkrec-xvect-voxceleb")
TTS_ID       = os.getenv("TTS_MODEL_ID", "microsoft/speecht5_tts")
VOC_ID       = os.getenv("VOCODER_ID",  "microsoft/speecht5_hifigan")
LM_ID        = os.getenv("LM_ID", "gpt2")
PACK_MODE    = os.getenv("PACK_MODE", "lm").lower()
XTYPE        = os.getenv("XTYPE", "fp16").lower()
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTN_IMPL = os.getenv("LLM_ATTN_IMPL")
QUANT     = os.getenv("LLM_QUANT", "").lower()
BATCH     = max(1, int(os.getenv("LLM_BATCH", "1")))
SEQ_LEN   = 1023

_b2u = bytes_to_unicode(); _u2b = {v: k for k, v in _b2u.items()}
bytes2text = lambda bs: ''.join(_b2u[b] for b in bs)
text2bytes = lambda txt: bytes(_u2b[c] for c in txt)

@lru_cache(maxsize=1)
def _load_asr():
    return whisper.load_model(WHISPER_ID, device=DEVICE)

@lru_cache(maxsize=1)
def _load_spk():
    return EncoderClassifier.from_hparams(source=SPK_ID, run_opts={"device": DEVICE})

@lru_cache(maxsize=1)
def _load_tts():
    proc = SpeechT5Processor.from_pretrained(TTS_ID)
    tts  = SpeechT5ForTextToSpeech.from_pretrained(TTS_ID).to(DEVICE)
    voc  = SpeechT5HifiGan.from_pretrained(VOC_ID).to(DEVICE)
    return proc, tts, voc

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
    if QUANT == "8bit":
        kw["load_in_8bit"] = True
    elif QUANT == "4bit":
        kw["load_in_4bit"] = True
    lm = AutoModelForCausalLM.from_pretrained(LM_ID, **kw).eval()
    return tok, lm

def encode(wav: np.ndarray, sr: int) -> Dict[str, Any]:
    asr = _load_asr()
    wav_t = torch.from_numpy(wav).to(DEVICE)
    if sr != 16_000:
        wav_t = torchaudio.functional.resample(wav_t, sr, 16_000)
    text = asr.transcribe(audio=wav_t, fp16=(DEVICE.type == "cuda"))['text'].strip()

    spk = _load_spk()
    with torch.no_grad():
        emb = spk.encode_batch(wav_t.unsqueeze(0)).squeeze(1)
    if XTYPE == "fp16":
        emb_bytes = emb.half().cpu().numpy().tobytes()
    elif XTYPE == "int8":
        emb_bytes = (emb.cpu().numpy() * 127).astype("int8").tobytes()
    else:
        emb_bytes = emb.cpu().numpy().tobytes()

    tok, _ = _load_lm()
    tokens = tok.encode(text, add_special_tokens=False)

    return {
        "tokens": tokens,
        "emb": emb_bytes,
        "dtype": XTYPE,
        "text_len": len(tokens),
    }

def pack(info: Dict[str, Any]) -> bytes:
    tok_ids = info["tokens"]
    tok, lm = _load_lm(); EOS = tok.eos_token_id

    enc = constriction.stream.queue.RangeEncoder()
    if PACK_MODE == "uniform":
        vocab = lm.config.vocab_size
        uni = np.full(vocab, 1 / vocab, dtype=np.float32)
        for tid in tok_ids:
            enc.encode(tid, constriction.stream.model.Categorical(uni, perfect=False))
    else:
        blocks = [tok_ids[i:i + SEQ_LEN] for i in range(0, len(tok_ids), SEQ_LEN)]
        ctx_prev = [EOS]
        with torch.inference_mode():
            for blk in blocks:
                inp = torch.tensor([ctx_prev + blk], device=DEVICE)
                probs = torch.softmax(lm(inp).logits[0, -len(blk):].float(), -1).cpu().numpy()
                for pos, tid in enumerate(blk):
                    enc.encode(tid, constriction.stream.model.Categorical(
                        probs[pos].astype(np.float32), perfect=False))
                ctx_prev = blk[-1:]

    text_stream_np = enc.get_compressed()
    text_bytes = text_stream_np.tobytes() if isinstance(text_stream_np, np.ndarray) else bytes(text_stream_np)

    header  = struct.pack("<I", len(text_bytes))
    payload = header + text_bytes + info["emb"]
    return payload

def decode(info: Dict[str, Any], sr: int) -> np.ndarray:
    emb_bytes = info["emb"]
    if info["dtype"] == "fp16":
        emb = torch.frombuffer(emb_bytes, dtype=torch.float16).view(1, 512).to(DEVICE)
    elif info["dtype"] == "int8":
        emb = (torch.frombuffer(emb_bytes, dtype=torch.int8).float() / 127).view(1, 512).to(DEVICE)
    else:
        emb = torch.frombuffer(emb_bytes, dtype=torch.float32).view(1, 512).to(DEVICE)

    tok, _ = _load_lm()
    text = tok.decode(info["tokens"])

    proc, tts, voc = _load_tts()
    ids = proc(text=text, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        wav_16k = tts.generate_speech(ids, speaker_embeddings=emb, vocoder=voc)
    return wav_16k.cpu().numpy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("SpeechT5 compression smoke test")
    parser.add_argument("wav")
    parser.add_argument("--outdir")
    a = parser.parse_args()

    wav, sr0 = sf.read(a.wav, dtype="float32", always_2d=False)
    info = encode(wav, sr0)
    payload = pack(info)
    recon = decode(info, sr0)

    outdir = Path(a.outdir) if a.outdir else Path(tempfile.mkdtemp(prefix="ttswrap_"))
    outdir.mkdir(parents=True, exist_ok=True)
    sf.write(outdir / "recon.wav", recon, 16_000)
    (outdir / "payload.bin").write_bytes(payload)

    kbps = 8 * len(payload) / (len(wav) / sr0) / 1000
    print(f"✓ {len(payload)} bytes ({kbps:.2f} kbps) → {outdir}")
