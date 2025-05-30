#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, importlib, json, multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    return wav, sr


def _job(args) -> Dict:
    codec_mod, wav_path, metrics_mod = args
    codec = importlib.import_module(codec_mod)
    metr  = importlib.import_module(metrics_mod)

    ref, sr = load_wav(wav_path)
    bitstream = codec.encode(ref, sr)
    deg = codec.decode(bitstream, sr)

    row = metr.compare(ref, deg, sr)
    row.update(codec=codec_mod, wav=wav_path)
    return row


def _aggregate(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("codec", "wav")]
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser("Batch evaluation of speech codecs")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSON list produced by prepare_datasets.py")
    ap.add_argument("--codec", required=True, nargs="+",
                    help="Import path(s) to codec wrappers with encode()/decode().")
    ap.add_argument("--out_csv", required=True, type=Path,
                    help="Destination CSV for per-file results.")
    ap.add_argument("--metrics_module", default="eval_metrics",
                    help="Module exposing compare(ref, deg, sr).")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="0/1 ⇒ run serially (no multiprocessing).")
    args = ap.parse_args()

    clips = json.loads(args.manifest.read_text())
    jobs  = [(c, item["wav"], args.metrics_module)
             for c in args.codec for item in clips]

    print(f"{len(jobs)} clip-codec pairs to process …")

    rows: List[Dict] = []

    if args.num_workers and args.num_workers > 1:
        # multiprocessing branch
        with mp.Pool(args.num_workers) as pool:
            for r in tqdm(pool.imap_unordered(_job, jobs, chunksize=16),
                          total=len(jobs), desc="eval", unit="pair"):
                rows.append(r)
    else:
        for j in tqdm(jobs, total=len(jobs), desc="eval", unit="pair"):
            rows.append(_job(j))

    keys = ["codec", "wav"] + [k for k in rows[0] if k not in ("codec", "wav")]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Per-file CSV →", args.out_csv)

    for c in args.codec:
        agg = _aggregate([r for r in rows if r["codec"] == c])
        summary = " | ".join(f"{k}:{v:.3f}" for k, v in agg.items())
        print(f"{c:<30} {summary}")


if __name__ == "__main__":
    main()
