#!/usr/bin/env python3
"""
batch_rate_eval.py – compute size (bytes, kbps, bits/token) and quality.
"""
from __future__ import annotations
import argparse, csv, importlib, json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

def load_wav(path: str):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    return wav, sr

def one_job(codec_mod: str, wav_path: str, metr_mod: str) -> Dict[str, Any]:
    codec = importlib.import_module(codec_mod)
    metr  = importlib.import_module(metr_mod)

    ref, sr = load_wav(wav_path)
    stream  = codec.encode(ref, sr)
    packed  = codec.pack(stream)        # <── NEW
    deg     = codec.decode(stream, sr)

    row = metr.compare(ref, deg, sr)    # PESQ/STOI/SDR/…
    row.update({
        "codec": codec_mod,
        "wav"  : wav_path,
        "bytes": len(packed),
        "kbps" : 8*len(packed)/(sr*len(ref)/sr)/1000,  # 8-bit per sec /1000
    })
    if isinstance(stream, (list, np.ndarray)):
        row["bpt"] = 8*len(packed)/len(stream)    # bits per token
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--codec",    required=True, nargs="+")
    ap.add_argument("--out_csv",  required=True, type=Path)
    ap.add_argument("--metrics_module", default="eval_metrics")
    a = ap.parse_args()

    clips = json.loads(a.manifest.read_text())
    jobs  = [(c, item["wav"], a.metrics_module)
             for c in a.codec for item in clips]

    rows: List[Dict] = []
    for j in tqdm(jobs, desc="eval", unit="pair"):
        rows.append(one_job(*j))

    keys = list(rows[0].keys())
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    print("CSV written to", a.out_csv)

if __name__ == "__main__":
    main()
