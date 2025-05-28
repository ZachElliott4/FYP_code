#!/usr/bin/env python3
"""
Benchmark classical audio codecs (FLAC, MP3, AAC, Opus) plus raw-PCM gzip/xz.

Example
-------
python scripts/traditional_codecs.py \
       --input  samples/ObamaSpeech.wav \
       --outdir results/trad_codecs
"""
from __future__ import annotations
import argparse, json, subprocess, gzip, lzma, contextlib, wave
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
#                               helpers                                        #
# --------------------------------------------------------------------------- #
def ffmpeg_encode(src: Path, dst: Path, ffmpeg_args: list[str]) -> int:
    """Encode *src* → *dst* with given ffmpeg CLI args, return bytes written."""
    if dst.exists():
        dst.unlink()
    cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(src), *ffmpeg_args, str(dst)]
    subprocess.run(cmd, check=True)
    return dst.stat().st_size


def pcm_from_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load int16 mono PCM & sampling-rate."""
    pcm, sr = sf.read(wav_path, dtype="int16")
    if pcm.ndim == 2:        # stereo → mono
        pcm = pcm.mean(axis=1).astype("int16")
    return pcm, sr


# --------------------------------------------------------------------------- #
#                                main                                          #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="16-kHz mono WAV to benchmark.")
    p.add_argument("--outdir", type=Path, default=Path("results/trad_codecs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    wav_path: Path = args.input
    assert wav_path.exists(), f"Missing {wav_path}"

    # --- basic info ---------------------------------------------------------
    with contextlib.closing(wave.open(str(wav_path), "rb")) as w:
        sr, nframes = w.getframerate(), w.getnframes()
    duration_s = nframes / sr
    wav_bytes  = wav_path.stat().st_size

    # --- codec spec list ----------------------------------------------------
    spec = [  # id, ffmpeg args, ext, lossless?
        ("flac",      ["-c:a", "flac"],                                        ".flac",  True),
        ("mp3_128k",  ["-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1"],      ".mp3",   False),
        ("aac_96k",   ["-c:a", "aac", "-b:a", "96k", "-ac", "1"],              ".aac",   False),
        ("opus_24k",  ["-c:a", "libopus", "-b:a", "24k", "-ac", "1",
                       "-application", "audio"],                               ".opus",  False),
    ]

    results: dict[str, int] = {}

    # --- ffmpeg codecs ------------------------------------------------------
    for cid, ffargs, ext, _ in tqdm(spec, desc="FFmpeg encodes"):
        dst = args.outdir / f"{wav_path.stem}_{cid}{ext}"
        results[cid] = ffmpeg_encode(wav_path, dst, ffargs)

    # --- raw-PCM + gzip / xz -------------------------------------------------
    pcm, _ = pcm_from_wav(wav_path)
    pcm_bytes = pcm.tobytes()

    gzip_path = args.outdir / f"{wav_path.stem}_raw_pcm.gz"
    with gzip.open(gzip_path, "wb", compresslevel=6) as f:
        f.write(pcm_bytes)
    results["gzip_raw"] = gzip_path.stat().st_size

    xz_path = args.outdir / f"{wav_path.stem}_raw_pcm.xz"
    with lzma.open(xz_path, "wb", preset=9 | lzma.PRESET_EXTREME) as f:
        f.write(pcm_bytes)
    results["lzma_raw"] = xz_path.stat().st_size

    # --- summary ------------------------------------------------------------
    summary: dict[str, dict[str, float | int]] = {}
    for k, b in results.items():
        summary[k] = {
            "bytes": b,
            "ratio_vs_wav": round(b / wav_bytes, 4),
            "kbps": round(8 * b / (1000 * duration_s), 2),
        }
    summary["wav_orig"] = {
        "bytes": wav_bytes,
        "ratio_vs_wav": 1.0,
        "kbps": round(8 * wav_bytes / (1000 * duration_s), 2),
    }

    out_json = args.outdir / "codec_sizes.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results → {out_json}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
