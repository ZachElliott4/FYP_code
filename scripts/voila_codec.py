from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch, torchaudio, constriction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoFeatureExtractor,
    AutoModel,
)

# Local helper
from transformers import AutoProcessor, EncodecModel

class VoilaTokenizer:
    def __init__(self,
                 model_name: str = "maitrix-org/Voila-Tokenizer",
                 bandwidth: float = 1.1,
                 device: str = "cpu") -> None:
        import torch
        self.device = torch.device(device)
        self.model = EncodecModel.from_pretrained(model_name).to(self.device)
        self.sample_rate = AutoProcessor.from_pretrained(model_name).sampling_rate
        self.bandwidth = bandwidth

    @torch.no_grad()
    def encode(self, wav):
        wav = wav.to(self.device).unsqueeze(0).unsqueeze(0)
        codes = self.model.encode(wav, bandwidth=self.bandwidth).audio_codes
        return codes[0, 0]

    @torch.no_grad()
    def decode(self, codes2d):
        """codes2d LongTensor [n_q, T] on self.device → 1-D float32 CPU wav"""
        wav = self.model.decode(codes2d[None, None], [None])[0]
        return wav[0, 0].cpu()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Path to 16-kHz mono WAV")
    p.add_argument("--outdir", type=Path, default=Path("outputs/voila"),
                   help="Directory to write recon.wav and stats.json")
    p.add_argument("--lm_id", default="maitrix-org/Voila-base")
    p.add_argument("--codec_id", default="maitrix-org/Voila-Tokenizer")
    p.add_argument("--trace_steps", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"device = {args.device}")

    # 1) Load models
    tok = AutoTokenizer.from_pretrained(args.lm_id)
    lm = AutoModelForCausalLM.from_pretrained(
        args.lm_id,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device).eval()

    tokenizer = VoilaTokenizer(args.codec_id, device=args.device)
    fx = AutoFeatureExtractor.from_pretrained(args.codec_id)
    codec = AutoModel.from_pretrained(args.codec_id).to(args.device).eval()

    # 2) Load audio
    wav, sr = torchaudio.load(str(args.input))
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    wav = wav.mean(0)  # mono
    orig_len = wav.numel()
    print(f"audio length {orig_len/16000:.2f}s")

    # 3) Tokenise with Voila tokenizer
    with torch.no_grad():
        enc_out = codec.encode(
            fx(wav, sampling_rate=16_000, return_tensors="pt")["input_values"].to(
                args.device
            )
        )
    codes_raw = enc_out.audio_codes
    if isinstance(codes_raw, torch.Tensor):
        n_q = codes_raw.shape[1]
        tokens = codes_raw.flatten().tolist()
    else:  # multi-track output (not expected here)
        n_q = len(codes_raw)
        tokens = [x.item() for track in codes_raw for x in track.squeeze(0)]

    print(f"tokenised → {len(tokens)} IDs  (codebooks={n_q})")

    # 4) Arithmetic-encode using the LM
    ids = [tok.eos_token_id] + tokens
    enc = constriction.stream.queue.RangeEncoder()

    with torch.inference_mode():  # <-- prevents grad + saves RAM
        for i in range(1, len(ids)):
            ctx = torch.tensor(ids[:i], device=args.device).unsqueeze(0)
            logits = lm(ctx).logits[0, -1]
            probs = torch.softmax(logits.float(), -1).cpu().numpy()
            if i <= args.trace_steps:
                top = probs.argsort()[-5:][::-1]
                print(f"step {i}: choose id {ids[i]}  (top-1 prob={probs[ids[i]]:.3e})")
            enc.encode(
                ids[i],
                constriction.stream.model.Categorical(probs.astype(np.float32), perfect=False),
            )

    bitstream = enc.get_compressed()
    bpt_lm = 8 * len(bitstream) / len(tokens)
    print(f"LM coding → {len(bitstream)} bytes  ({bpt_lm:.2f} bits/token)")

    # 5) Uniform baseline
    vocab = lm.config.vocab_size
    uniform = np.ones(vocab, dtype=np.float32) / vocab
    enc_u = constriction.stream.queue.RangeEncoder()
    for t in tokens:
        enc_u.encode(
            t, constriction.stream.model.Categorical(uniform, perfect=False)
        )
    bitstream_u = enc_u.get_compressed()
    bpt_u = 8 * len(bitstream_u) / len(tokens)
    print(
        f"Uniform  → {len(bitstream_u)} bytes  ({bpt_u:.2f} bits/token)  "
        f"gain {bpt_u - bpt_lm:.2f}  ({(bpt_u-bpt_lm)/bpt_u:.1%})"
    )

    # 6) Decode & verify
    dec, decoded = constriction.stream.queue.RangeDecoder(bitstream), []
    with torch.inference_mode():
        for _ in range(1, len(ids)):
            ctx = torch.tensor([tok.eos_token_id] + decoded, device=args.device).unsqueeze(0)
            probs = torch.softmax(lm(ctx).logits[0, -1].float(), -1).cpu().numpy()
            decoded.append(
                dec.decode(constriction.stream.model.Categorical(probs.astype(np.float32),perfect=False))
            )
    assert decoded == tokens, "round-trip mismatch"
    print("round-trip verification ✓")

    # 7) Reconstruct waveform
    codes2d = torch.tensor(decoded, dtype=torch.long, device=args.device).view(n_q, -1)
    recon = tokenizer.decode(codes2d)[:orig_len].unsqueeze(0)
    out_wav = args.outdir / "recon.wav"
    torchaudio.save(str(out_wav), recon, 16_000)

    # 8) Save stats
    stats = {
        "n_tokens": len(tokens),
        "bits_per_token_lm": bpt_lm,
        "bits_per_token_uniform": bpt_u,
    }
    (args.outdir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("done → files in", args.outdir)


if __name__ == "__main__":
    main()
