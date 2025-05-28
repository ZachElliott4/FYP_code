#!/usr/bin/env python3
# scripts/voila_codec.py
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch, torchaudio, constriction
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, EncodecModel

class VoilaTokenizer:
    def __init__(
        self,
        model_name: str = "maitrix-org/Voila-Tokenizer",
        bandwidth: float = 1.1,           # --codec_bw
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = EncodecModel.from_pretrained(model_name).to(self.device)
        self.sample_rate = AutoProcessor.from_pretrained(model_name).sampling_rate
        self.bandwidth = bandwidth

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav 1-D (N) or 2-D (1, N) → LongTensor [n_q, T] on self.device"""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device).unsqueeze(0)          # (1, 1, N)
        codes = self.model.encode(wav, bandwidth=self.bandwidth).audio_codes
        return codes[0, 0]                              # (n_q, T)

    @torch.no_grad()
    def decode(self, codes2d: torch.Tensor) -> torch.Tensor:
        """codes2d LongTensor [n_q, T] → mono float32 wav (1-D cpu)"""
        wav = self.model.decode(codes2d[None, None], [None])[0]   # (1, 1, N)
        return wav[0, 0].cpu()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Path to 16-kHz mono WAV")
    p.add_argument("--outdir", type=Path, default=Path("outputs/voila"),
                   help="Directory that will receive recon.wav & stats.json")
    p.add_argument("--lm_id",    default="maitrix-org/Voila-base")
    p.add_argument("--codec_id", default="maitrix-org/Voila-Tokenizer")
    p.add_argument("--codec_bw", type=float, default=1.1,
                   help="Bandwidth (kb/s) to use for encode & decode")
    p.add_argument("--trace_steps", type=int, default=4,
                   help="Print LM probs for the first N tokens")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"device = {args.device}  |  bandwidth = {args.codec_bw} kb/s")

    # 1) Load LM
    tok = AutoTokenizer.from_pretrained(args.lm_id)
    lm  = AutoModelForCausalLM.from_pretrained(
            args.lm_id,
            torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(args.device).eval()

    # 2) Load Voila tokenizer (encode & decode share one obj)
    tokenizer = VoilaTokenizer(args.codec_id,
                               bandwidth=args.codec_bw,
                               device=args.device)

    # 3) Read audio
    wav, sr = torchaudio.load(str(args.input))
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    wav = wav.mean(0)                                     # mono
    orig_len = wav.numel()
    print(f"audio length = {orig_len/16000:.2f} s")

    # 4) Encode to discrete units
    codes2d = tokenizer.encode(wav)                       # (n_q, T)
    n_q     = codes2d.shape[0]
    tokens  = codes2d.flatten().tolist()
    print(f"tokenised → {len(tokens)} IDs  (codebooks = {n_q})")

    # 5) Arithmetic-encode with LM
    ids   = [tok.eos_token_id] + tokens                   # prefix EOS
    enc   = constriction.stream.queue.RangeEncoder()

    with torch.inference_mode():
        for i in range(1, len(ids)):
            ctx    = torch.tensor(ids[:i], device=args.device).unsqueeze(0)
            probs  = torch.softmax(lm(ctx).logits[0, -1].float(), -1).cpu().numpy()
            if i <= args.trace_steps:
                print(f"step {i:>3}: id={ids[i]:>5}  p={probs[ids[i]]:.3e}")
            enc.encode(ids[i],
                       constriction.stream.model.Categorical(
                           probs.astype(np.float32), perfect=False))

    bitstream = enc.get_compressed()
    bpt_lm    = 8 * len(bitstream) / len(tokens)
    print(f"LM coding → {len(bitstream)} bytes  ({bpt_lm:.2f} bits/token)")

    # 6) Uniform baseline
    vocab     = lm.config.vocab_size
    uniform   = np.full(vocab, 1.0 / vocab, dtype=np.float32)
    enc_u     = constriction.stream.queue.RangeEncoder()
    for t in tokens:                                      # same symbol set
        enc_u.encode(t, constriction.stream.model.Categorical(uniform, perfect=False))
    bitstream_u = enc_u.get_compressed()
    bpt_u       = 8 * len(bitstream_u) / len(tokens)
    print(f"Uniform  → {len(bitstream_u)} bytes  ({bpt_u:.2f} bits/token)  "
          f"gain {bpt_u - bpt_lm:.2f}  ({(bpt_u - bpt_lm)/bpt_u:.1%})")

    # 7) Decode & verify round-trip
    dec, decoded = constriction.stream.queue.RangeDecoder(bitstream), []
    with torch.inference_mode():
        for _ in range(len(tokens)):
            ctx   = torch.tensor([tok.eos_token_id] + decoded,
                                 device=args.device).unsqueeze(0)
            probs = torch.softmax(lm(ctx).logits[0, -1].float(), -1).cpu().numpy()
            decoded.append(dec.decode(
                constriction.stream.model.Categorical(probs.astype(np.float32),
                                                      perfect=False)))
    assert decoded == tokens, "round-trip mismatch"
    print("round-trip verification ✓")

    # 8) Reconstruct waveform
    recon = tokenizer.decode(torch.tensor(decoded, dtype=torch.long,
                                          device=args.device).view(n_q, -1))
    recon = recon[:orig_len].unsqueeze(0).clamp_(-1.0, 1.0)
    torchaudio.save(str(args.outdir / "recon.wav"), recon, 16_000)

    # 9) Save stats
    (args.outdir / "stats.json").write_text(json.dumps({
        "n_tokens":           len(tokens),
        "bits_per_token_lm":  bpt_lm,
        "bits_per_token_uni": bpt_u,
        "bandwidth_kbps":     args.codec_bw
    }, indent=2), encoding="utf-8")

    print("done → files in", args.outdir)


if __name__ == "__main__":
    main()
