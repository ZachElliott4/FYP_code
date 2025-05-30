# scripts/prepare_datasets.py
from __future__ import annotations
import argparse, json, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import librosa
import datasets
from tqdm import tqdm

DATASETS: Dict[str, Dict] = {
    "librispeech_clean": {
        "hf_name":  "librispeech_asr",
        "hf_config": "clean",
        "hf_split":  "test",
    },
    "librispeech_other": {
        "hf_name":  "librispeech_asr",
        "hf_config": "other",
        "hf_split":  "test",
    },
    "commonvoice_en": {                         # Common Voice v11 EN test
        "hf_name":  "mozilla-foundation/common_voice_11_0",
        "hf_config": "en",
        "hf_split":  "test",
        "trust_remote_code": True,              # suppress prompt
    },
    "voxpopuli_en": {                           # VoxPopuli EN test
        "hf_name":  "facebook/voxpopuli",
        "hf_config": "en",
        "hf_split":  "test",
        "trust_remote_code": True,              # suppress prompt
    },
}

TARGET_SR  = 16_000
MAX_SEC    = 30
PCM_MAX    = 32_767
QUOTA_MB   = 250                               # per corpus
rng        = random.Random()

def resample_mono(wav: np.ndarray, sr: int) -> np.ndarray:
    """Return ≤30 s mono int16 at 16 kHz."""
    if wav.ndim == 2 and wav.shape[0] > 1:
        wav = wav.mean(axis=0)
    if sr != TARGET_SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    wav = wav[: MAX_SEC * TARGET_SR] if len(wav) > MAX_SEC * TARGET_SR else wav
    wav = np.clip(wav * PCM_MAX if wav.dtype != np.int16 else wav, -PCM_MAX, PCM_MAX)
    return wav.astype(np.int16)

def dir_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*.wav"))

def next_counter(folder: Path) -> int:
    nums = [int(p.stem.split("_")[-1]) for p in folder.glob("*.wav")
            if p.stem.split("_")[-1].isdigit()]
    return max(nums, default=0) + 1

def load_audio_field(audio_field) -> Tuple[np.ndarray, int]:
    """Handle decoded dicts returned by 🤗 `datasets`."""
    if isinstance(audio_field, dict) and audio_field.get("array") is not None:
        return np.asarray(audio_field["array"]), audio_field["sampling_rate"]
    raise ValueError("Unexpected audio field structure")

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download ≈250 MB per English corpus and build eval_manifest.json"
    )
    ap.add_argument("--target_dir",   required=True, type=Path)
    ap.add_argument("--manifest_out", required=True, type=Path)
    ap.add_argument("--per_dataset_mb", type=int, default=QUOTA_MB)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Delete target_dir first, then rebuild.")
    args = ap.parse_args()

    quota_bytes = args.per_dataset_mb * 1_048_576
    rng.seed(args.seed)

    if args.force and args.target_dir.exists():
        shutil.rmtree(args.target_dir)
    args.target_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []

    for tag, meta in DATASETS.items():
        print(f"\n=== {tag} ===")
        c_dir = args.target_dir / tag
        c_dir.mkdir(parents=True, exist_ok=True)

        bytes_have = dir_bytes(c_dir)
        print(f" already present: {bytes_have/1e6:.1f} MB")
        if bytes_have >= quota_bytes:
            print(" quota satisfied — skipping")
            for wav in c_dir.glob("*.wav"):
                manifest.append(
                    {"wav": str(wav), "dataset": tag, "bytes": wav.stat().st_size}
                )
            continue

        need = quota_bytes - bytes_have
        print(f" need ~{need/1e6:.1f} MB")
        counter = next_counter(c_dir)

        ds = datasets.load_dataset(
            meta["hf_name"],
            meta["hf_config"],
            split=meta["hf_split"],
            streaming=True,
            trust_remote_code=meta.get("trust_remote_code", False),
        ).shuffle(buffer_size=10_000, seed=args.seed)

        for ex in tqdm(ds, desc=tag):
            wav, sr = load_audio_field(ex["audio"])
            wav = resample_mono(wav, sr)

            out = c_dir / f"{tag}_{counter:06d}.wav"
            counter += 1

            sf.write(out, wav, TARGET_SR, subtype="PCM_16")
            sz = out.stat().st_size
            bytes_have += sz
            manifest.append({"wav": str(out), "dataset": tag, "bytes": sz})

            if bytes_have >= quota_bytes:
                break
        print(f" final size: {bytes_have/1e6:.1f} MB")

    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    total_mb = sum(m["bytes"] for m in manifest) / 1e6
    print(f"\n✓ Done — {len(manifest)} files, {total_mb:.1f} MB total.")
    print("Manifest saved to", args.manifest_out)


if __name__ == "__main__":
    main()
