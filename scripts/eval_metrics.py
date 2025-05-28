#!/usr/bin/env python3
"""
metrics_eval.py  –  objective metrics for speech‐codec evaluation
────────────────────────────────────────────────────────────────
USAGE
-----
    python scripts/metrics_eval.py \
           --ref_path samples/ObamaSpeech.wav \
           --deg_path outputs/voila_demo/recon.wav \
           --outfile  outputs/metrics.json

If --outfile is omitted, results are just printed.

Implemented metrics
-------------------
1. PESQ-WB (MOS-LQO)         – ITU-T P.862.2, wide-band
2. STOI                      – Short-Time Objective Intelligibility
3. SI-SDR   [dB]             – Scale-Invariant Signal-to-Distortion Ratio
4. Log-Spectral Distance     – mean Euclidean distance on log-magnitude STFT
5. Speaker-Embedding CosSim  – ECAPA-TDNN cosine similarity  (optional)
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

have_pesq = have_stoi = have_resem = True
try:
    from pesq import pesq
except ModuleNotFoundError:
    have_pesq = False
    warnings.warn("`pesq` package not found ⇒ PESQ not computed.")
try:
    from pystoi import stoi
except ModuleNotFoundError:
    have_stoi = False
    warnings.warn("`pystoi` package not found ⇒ STOI not computed.")
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
except ModuleNotFoundError:
    have_resem = False
    warnings.warn("`resemblyzer` package not found ⇒ speaker-embed cosine sim not computed.")



def load_mono(path: Path, target_sr: int = 16_000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


def si_sdr(ref: np.ndarray, deg: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-Invariant SDR (Le Roux + Wisdom, 2019)."""
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    alpha = np.dot(deg, ref) / (np.dot(ref, ref) + eps)
    proj = alpha * ref
    noise = deg - proj
    return 10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps))


def log_spectral_distance(ref: np.ndarray, deg: np.ndarray,
                          n_fft: int = 1024, hop: int = 256) -> float:
    """Mean Euclidean distance between log-magnitude spectra (lower = better)."""
    R = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)) + 1e-10
    D = np.abs(librosa.stft(deg, n_fft=n_fft, hop_length=hop)) + 1e-10
    lsd = np.linalg.norm(20 * np.log10(R) - 20 * np.log10(D), axis=0)
    return lsd.mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Objective metrics for speech codecs")
    p.add_argument("--ref_path", type=Path, required=True, help="clean reference WAV")
    p.add_argument("--deg_path", type=Path, required=True, help="degraded / recon WAV")
    p.add_argument("--outfile", type=Path, help="optional JSON output file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ref = load_mono(args.ref_path)
    deg = load_mono(args.deg_path)
    L = min(len(ref), len(deg))
    if len(ref) != len(deg):
        warnings.warn("Reference and degraded signals have different lengths; "
                      "truncating to shortest.")
    ref, deg = ref[:L], deg[:L]

    metrics: dict[str, float | None] = {}

    if have_pesq:
        metrics["pesq_wb"] = float(pesq(16_000, ref, deg, "wb"))
    else:
        metrics["pesq_wb"] = None

    if have_stoi:
        metrics["stoi"] = float(stoi(ref, deg, 16_000, extended=False))
    else:
        metrics["stoi"] = None

    metrics["si_sdr"] = float(si_sdr(ref, deg))

    metrics["log_spectral_distance"] = float(log_spectral_distance(ref, deg))

    if have_resem:
        emb_ref = encoder.embed_utterance(preprocess_wav(ref, 16_000))
        emb_deg = encoder.embed_utterance(preprocess_wav(deg, 16_000))
        metrics["spk_emb_cossim"] = float(np.dot(emb_ref, emb_deg) /
                                          (np.linalg.norm(emb_ref) *
                                           np.linalg.norm(emb_deg)))
    else:
        metrics["spk_emb_cossim"] = None

    print("\n───────── Objective Metrics ─────────")
    for k, v in metrics.items():
        print(f"{k:23s} : {v:.4f}" if v is not None else f"{k:23s} : n/a")
    print("──────────────────────────────────────")

    if args.outfile:
        args.outfile.parent.mkdir(parents=True, exist_ok=True)
        args.outfile.write_text(json.dumps(metrics, indent=2))
        print("Saved JSON →", args.outfile)


if __name__ == "__main__":
    main()
