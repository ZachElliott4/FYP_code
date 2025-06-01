from __future__ import annotations

import argparse, csv, importlib, json, multiprocessing as mp, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

from tqdm.auto import tqdm

def _load_wav(path: str):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=0)
    return wav, sr


def _one_job(job) -> Dict[str, Any]:
    codec_mod, wav_path, metr_mod = job

    codec = importlib.import_module(codec_mod)
    metr  = importlib.import_module(metr_mod)

    ref, sr = _load_wav(wav_path)
    bitstream = codec.encode(ref, sr)
    packed    = codec.pack(bitstream)
    deg       = codec.decode(bitstream, sr)

    row: Dict[str, Any] = metr.compare(ref, deg, sr)
    row.update({
        "codec" : codec_mod,
        "wav"   : wav_path,
        "bytes" : len(packed),
        "kbps"  : 8 * len(packed) / (len(ref) / sr) / 1000,
    })
    codes = bitstream.get("codes")
    if codes is not None:
        n_tokens = codes.numel() if hasattr(codes, "numel") else len(codes)
        row["bpt"] = 8 * len(packed) / n_tokens
    return row

def main():
    ap = argparse.ArgumentParser(description="Batch objective evaluation of speech codecs.")
    ap.add_argument("--manifest", required=True, type=Path,
                    help="JSON list of {'wav': path, ...} objects")
    ap.add_argument("--codec",    required=True, nargs="+",
                    help="Python module(s) exposing encode/pack/decode")
    ap.add_argument("--out_csv",  required=True, type=Path,
                    help="Where to write the per‑clip CSV results")
    ap.add_argument("--metrics_module", default="eval_metrics",
                    help="Module with compare(ref, deg, sr) → dict")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="0 = synchronous main‑process; >=1 uses multiprocessing")
    args = ap.parse_args()

    clips = json.loads(args.manifest.read_text())
    if not clips:
        sys.exit("Manifest is empty – nothing to evaluate.")

    jobs = [(c, item["wav"], args.metrics_module) for c in args.codec for item in clips]

    rows: List[Dict[str, Any]] = []

    ctx_name = "fork" if os.name != "nt" and args.num_workers else "spawn"
    ctx = mp.get_context(ctx_name)

    if args.num_workers <= 1:
        for j in tqdm(jobs, desc="eval", unit="pair"):
            rows.append(_one_job(j))
    else:
        with ctx.Pool(args.num_workers) as pool, \
             tqdm(total=len(jobs), desc="eval", unit="pair") as pbar:
            for r in pool.imap_unordered(_one_job, jobs, chunksize=8):
                rows.append(r); pbar.update()

    keys = list(rows[0].keys())
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=keys).writeheader()
        csv.DictWriter(f, fieldnames=keys).writerows(rows)

    print(f"CSV written to {args.out_csv}  (rows = {len(rows)})")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total wall‑time: {time.time()-start:.1f} s")
