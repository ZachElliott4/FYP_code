from __future__ import annotations
import os
from functools import lru_cache
from typing import Any, Tuple

import numpy as np
import torch
from transformers import AutoProcessor, EncodecModel

CODEC_ID = os.getenv("VOILA_CODEC_ID", "maitrix-org/Voila-Tokenizer")
BANDWIDTH = float(os.getenv("VOILA_BW", 1.1))
DEVICE = torch.device(os.getenv("VOILA_DEVICE",
                                "cuda" if torch.cuda.is_available() else "cpu"))

@lru_cache(maxsize=1)
def _load_codec() -> Tuple[EncodecModel, int]:
    model = EncodecModel.from_pretrained(CODEC_ID).to(DEVICE).eval()
    sr = AutoProcessor.from_pretrained(CODEC_ID).sampling_rate
    return model, sr

def _wav_to_tensor(wav: np.ndarray) -> torch.Tensor:
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]            # (1, N)
    return torch.from_numpy(wav).float().to(DEVICE)

def encode(wav: np.ndarray, sr: int = 16_000) -> Any:
    model, codec_sr = _load_codec()
    if sr != codec_sr:
        wav = torch.from_numpy(wav).float()
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(0).unsqueeze(0), scale_factor=codec_sr/sr,
            mode="linear", align_corners=False).squeeze().numpy()
    wav_t = _wav_to_tensor(wav)
    with torch.inference_mode():
        codes = model.encode(wav_t.unsqueeze(0),
                             bandwidth=BANDWIDTH).audio_codes[0, 0]  # (n_q,T)
    return codes.cpu().numpy()          # batch_eval just passes this along

def decode(bitstream: Any, sr: int = 16_000) -> np.ndarray:
    model, codec_sr = _load_codec()
    codes = torch.from_numpy(bitstream).long().to(DEVICE)   # (n_q,T)
    with torch.inference_mode():
        wav_t = model.decode(codes[None, None], [None])[0][0, 0]  # (N)
    wav = wav_t.cpu().numpy()
    if sr != codec_sr:
        wav_t = torch.from_numpy(wav).float()
        wav_t = torch.nn.functional.interpolate(
            wav_t.unsqueeze(0).unsqueeze(0), scale_factor=sr/codec_sr,
            mode="linear", align_corners=False).squeeze()
        wav = wav_t.numpy()
    return np.clip(wav, -1.0, 1.0).astype("float32")        # (N,)
