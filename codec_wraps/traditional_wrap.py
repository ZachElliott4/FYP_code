#!/usr/bin/env python3
"""
Traditional codec
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf

_FF_SPECS: Dict[str, tuple[list[str], str]] = {
    "flac":      (["-c:a", "flac"],                                    ".flac"),
    "mp3_128k":  (["-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1"], ".mp3"),
    "aac_96k":   (["-c:a", "aac", "-b:a", "96k",  "-ac", "1"],     ".aac"),
    "opus_24k":  (["-c:a", "libopus", "-b:a", "24k", "-ac", "1",
                    "-application", "audio"],                            ".opus"),
}
_CODEC_ID = os.getenv("TRAD_CODEC", "opus_24k").lower()
_FF_ARGS, _EXT = _FF_SPECS.get(_CODEC_ID, _FF_SPECS["opus_24k"])

def _ffmpeg(src: Path, dst: Path, ffargs: list[str]) -> None:
    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-i", str(src),
        *ffargs,
        str(dst),
    ]
    subprocess.run(cmd, check=True)

def encode(wav: np.ndarray, sr: int) -> Dict[str, bytes]:
    with tempfile.TemporaryDirectory() as tmpdir:
        pcm = Path(tmpdir) / "in.wav"
        sf.write(pcm, wav.astype(np.float32), sr, subtype="PCM_16")
        comp = Path(tmpdir) / f"out{_EXT}"
        _ffmpeg(pcm, comp, _FF_ARGS)
        return {"bytes": comp.read_bytes(), "sr": sr}


def pack(bitstream: Dict[str, bytes]) -> bytes:
    return bitstream["bytes"]


def decode(bitstream: Dict[str, bytes], sr: int | None = None) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = Path(tmpdir) / f"in{_EXT}"
        comp.write_bytes(bitstream["bytes"])
        wav = Path(tmpdir) / "out.wav"
        _ffmpeg(comp, wav, ["-c:a", "pcm_s16le", "-ac", "1"])
        audio, _ = sf.read(wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)

if __name__ == "__main__":
    import argparse, tempfile, pathlib

    ap = argparse.ArgumentParser(description="Manual test for traditional codec wrapper.")
    ap.add_argument("wav", help="Path to an audio file (any format).")
    args = ap.parse_args()

    audio, sr0 = sf.read(args.wav, dtype="float32", always_2d=False)
    bitstream  = encode(audio, sr0)
    payload    = pack(bitstream)
    recon      = decode(bitstream, sr0)

    outdir = pathlib.Path(tempfile.mkdtemp(prefix="trad_"))
    sf.write(outdir / "recon.wav", recon, sr0)
    (outdir / f"compressed{_EXT}").write_bytes(payload)

    kbps = 8 * len(payload) / (len(audio)/sr0) / 1000
    print(f"✓ Round‑trip OK – {len(payload)} bytes ({kbps:.2f} kb/s). Files in {outdir}")
