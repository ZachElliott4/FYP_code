#!/usr/bin/env python3
"""
eval_metrics.py – objective metrics for speech‐codec evaluation
Implements PESQ-WB (optional), STOI, SI-SDR, Log-Spectral Distance,
and speaker-embedding cosine similarity.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import librosa

have_pesq = False
try:
    from pesq import pesq
    have_pesq = True
except Exception as e:
    warnings.warn(f"PESQ disabled ({e}).")

have_stoi = False
try:
    from pystoi import stoi
    have_stoi = True
except ModuleNotFoundError:
    warnings.warn("`pystoi` not found → STOI disabled.")

have_resem = False
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    have_resem = True
except ModuleNotFoundError:
    warnings.warn("`resemblyzer` not found → speaker-cosine disabled.")

def load_mono(path: Path, target_sr: int = 16_000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype("float32")

def si_sdr(ref: np.ndarray, deg: np.ndarray, eps: float = 1e-8) -> float:
    ref, deg = ref - ref.mean(), deg - deg.mean()
    alpha = np.dot(deg, ref) / (np.dot(ref, ref) + eps)
    proj, noise = alpha * ref, deg - alpha * ref
    return 10 * np.log10((np.sum(proj**2) + eps) / (np.sum(noise**2) + eps))

def log_spectral_distance(ref: np.ndarray, deg: np.ndarray,
                          n_fft: int = 1024, hop: int = 256) -> float:
    R = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)) + 1e-10
    D = np.abs(librosa.stft(deg, n_fft=n_fft, hop_length=hop)) + 1e-10
    return float(np.mean(np.linalg.norm(20*np.log10(R) - 20*np.log10(D), axis=0)))

def compare(ref: np.ndarray, deg: np.ndarray, sr: int = 16_000) -> Dict[str, float|None]:
    L = min(len(ref), len(deg))
    ref, deg = ref[:L], deg[:L]

    m: Dict[str, float|None] = {}

    if have_pesq:
        try:
            m["pesq"] = float(pesq(sr, ref, deg, "wb"))
        except Exception:
            m["pesq"] = None
    else:
        m["pesq"] = None

    if have_stoi:
        m["stoi"] = float(stoi(ref, deg, sr, extended=False))
    else:
        m["stoi"] = None

    m["sdr"] = float(si_sdr(ref, deg))
    m["lsd"] = float(log_spectral_distance(ref, deg))

    if have_resem:
        er = encoder.embed_utterance(preprocess_wav(ref, sr))
        ed = encoder.embed_utterance(preprocess_wav(deg, sr))
        m["spk_cos"] = float(np.dot(er, ed)/(np.linalg.norm(er)*np.linalg.norm(ed)))
    else:
        m["spk_cos"] = None
    return m

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_path", type=Path, required=True)
    ap.add_argument("--deg_path", type=Path, required=True)
    ap.add_argument("--outfile",  type=Path)
    a = ap.parse_args()

    ref = load_mono(a.ref_path)
    deg = load_mono(a.deg_path)
    out = compare(ref, deg)

    print(json.dumps(out, indent=2))
    if a.outfile:
        a.outfile.parent.mkdir(parents=True, exist_ok=True)
        a.outfile.write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    _cli()
