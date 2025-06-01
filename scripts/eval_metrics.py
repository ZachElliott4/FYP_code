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
    from resemblyzer import VoiceEncoder
    encoder = VoiceEncoder()
    have_resem = True
except ModuleNotFoundError:
    warnings.warn("`resemblyzer` not found → speaker-cosine disabled.")

try:
    import torchaudio
    _have_ta = True
except ModuleNotFoundError:
    _have_ta = False

def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if _have_ta:
        import torch
        return torchaudio.functional.resample(torch.from_numpy(x), sr_in, sr_out).numpy()
    return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out)

def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x

def load_mono(path: Path, target_sr: int = 16_000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    wav = _to_mono(wav)
    wav = _resample(wav, sr, target_sr)
    return wav.astype("float32")

def _scale_full(fs: np.ndarray) -> np.ndarray:
    pk = np.max(np.abs(fs))
    return fs / pk if pk > 0 else fs

def si_sdr(ref: np.ndarray, deg: np.ndarray, eps: float = 1e-8) -> float:
    ref, deg = ref - ref.mean(), deg - deg.mean()
    alpha = np.dot(deg, ref) / (np.dot(ref, ref) + eps)
    proj, noise = alpha * ref, deg - alpha * ref
    return 10 * np.log10((np.sum(proj**2) + eps) / (np.sum(noise**2) + eps))

def log_spectral_distance(ref: np.ndarray, deg: np.ndarray,
                          n_fft: int = 1024, hop: int = 256) -> float:
    R = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)) + 1e-10
    D = np.abs(librosa.stft(deg, n_fft=n_fft, hop_length=hop)) + 1e-10
    diff_db = 20*np.log10(R) - 20*np.log10(D)
    return float(np.sqrt(np.mean(diff_db**2)))

def compare(ref: np.ndarray, deg: np.ndarray, sr: int = 16_000) -> Dict[str, float|None]:
    L = min(len(ref), len(deg))
    ref, deg = ref[:L], deg[:L]

    m: Dict[str, float|None] = {}

    if have_pesq:
        try:
            m["pesq"] = float(pesq(sr, _scale_full(ref), _scale_full(deg), "wb"))
        except Exception:
            m["pesq"] = None
    else:
        m["pesq"] = None

    if have_stoi:
        m["stoi"] = float(stoi(_scale_full(ref), _scale_full(deg), sr, extended=False))
    else:
        m["stoi"] = None

    m["sdr"] = float(si_sdr(ref, deg))
    m["lsd"] = float(log_spectral_distance(ref, deg))

    if have_resem:
        er = encoder.embed_utterance(ref)
        ed = encoder.embed_utterance(deg)
        m["spk_cos"] = float(np.dot(er, ed) /
                             (np.linalg.norm(er) * np.linalg.norm(ed)))
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
