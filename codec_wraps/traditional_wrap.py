"""
The codec variant is chosen via env var TRAD_CODEC:
    export TRAD_CODEC=flac
    export TRAD_CODEC=mp3_128k
    export TRAD_CODEC=aac_96k
    export TRAD_CODEC=opus_24k
"""
from __future__ import annotations
import os, subprocess, tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

_FF_SPECS = {
    "flac":       (["-c:a", "flac"],                                ".flac"),
    "mp3_128k":   (["-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1"], ".mp3"),
    "aac_96k":    (["-c:a", "aac", "-b:a", "96k", "-ac", "1"],      ".aac"),
    "opus_24k":   (["-c:a", "libopus", "-b:a", "24k", "-ac", "1",
                    "-application", "audio"],                       ".opus"),
}
_CODEC_ID = os.getenv("TRAD_CODEC", "opus_24k").lower()
_FF_ARGS, _EXT = _FF_SPECS.get(_CODEC_ID, _FF_SPECS["opus_24k"])

def _ffmpeg(src: Path, dst: Path, ffargs: list[str]) -> None:
    cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(src), *ffargs, str(dst)]
    subprocess.run(cmd, check=True)

def encode(wav: np.ndarray, sr: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        pcm = Path(tmpdir) / "in.wav"
        sf.write(pcm, wav, sr, subtype="PCM_16")
        comp = Path(tmpdir) / f"out{_EXT}"
        _ffmpeg(pcm, comp, _FF_ARGS)
        return comp.read_bytes()


def decode(bitstream: bytes, sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        comp = Path(tmpdir) / f"in{_EXT}"
        comp.write_bytes(bitstream)
        wav = Path(tmpdir) / "out.wav"
        _ffmpeg(comp, wav, ["-c:a", "pcm_s16le", "-ac", "1"])
        audio, _ = sf.read(wav, dtype="float32")
        if audio.ndim == 2:                       # stereo safety
            audio = audio.mean(axis=1)
        return audio.astype("float32")
