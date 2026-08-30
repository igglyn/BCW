"""
Byte Conditioning Wave
=================================================
BCW: a byte compressor that takes in N byte positions and reduces into a single embedding
This uses a teacher/distill pipeline using shared weights between two passes:
 The (encoder/decoder, gate_head, context_proj, pos_embed, pad_embed) weights are shared,
 And (byte_embed, byte_head, lat_head) are kept seperate.

Compression mechanism:
  After encoding, each of the 256 positions is a learned mix of:
    content = content_proj(raw_cartesian(encoder_output))  [meaningful signal]
    padding = pad_embed                                    [learned constant]
  Mixed by a per-position gate g = sigmoid(gate_head(raw_cartesian(z))):
    gated = g * content + (1-g) * pad_embed
  Ratio = g.mean() ∈ (0,1) is the fraction of content used.
  Compression loss = lambda_compress * (ratio1 + ratio2) minimises this.
  Reconstruction losses (r1, r2) counterbalance: compress too much and
  quality collapses.  Equilibrium is the minimum ratio the model can
  afford given its reconstruction targets.

  Gradient path: g is directly in the differentiable gating operation,
  so d(loss)/d(gate_head) is always non-zero from compression loss and
  non-zero from reconstruction losses whenever content ≠ pad_embed.

Pipeline:
  pass 1: byte_embed(256) → encoder → gate → gated(256) → decoder → byte_head
  pass 2: gated(256)      → encoder → gate → gated(256) → decoder → lat_head

The second pass only exists for training, as it will be absorbed by the first

Losses:
  r1:       cross-entropy byte reconstruction (BLT1 task)
  r2:       MSE latent reconstruction (BLT2 task)
  compress: ratio1 + ratio2 (compression budget pressure)
  var/cov:  VICReg on gated sequences across batch×position

python main.py preprocess <text.txt> <cache_prefix>
python main.py train <cache_prefix> [steps] [batch_size] [workers]
"""

import sys

import torch

from BCW.bcw import BCW
from BCW.dataset import preprocess_bcw, make_loader
from BCW.train import run_training


# ── constants ──────────────────────────────────────────────────────────────
PAD_ID  = 256
VOCAB   = 257       # bytes 0-255 + PAD
D_MODEL = 4
LAYERS = 1

N_BYTES = 256      # fixed input/output sequence length
STEPS = 48000
BS = 64



def main() -> None:
    """
    python bcw_bench.py preprocess <text.txt> <cache_prefix>
    python bcw_bench.py train <cache_prefix> [steps] [batch_size] [workers]
    """
    if len(sys.argv) < 2:
        print(__doc__); return

    mode = sys.argv[1]

    if mode == "preprocess":
        text_path  = sys.argv[2] if len(sys.argv) > 2 else "data.txt"
        out_prefix = sys.argv[3] if len(sys.argv) > 3 else "bcw_cache"
        preprocess_bcw(text_path, out_prefix, N_BYTES)
        return

    if mode == "train":
        out_prefix  = sys.argv[2] if len(sys.argv) > 2 else "bcw_cache"
        num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    else:
        print(f"unknown mode: {mode}"); return

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tr_loader  = make_loader(out_prefix, "tr",  bs=BS,
                             num_workers=num_workers)
    val_loader = make_loader(out_prefix, "val", bs=BS,
                             shuffle=False, num_workers=num_workers)

    model = BCW(vocab_size=VOCAB, ctx_length=N_BYTES, d=D_MODEL, depth=LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BCW params: {n_params:,}")
    print(f"  shared — encoder: {sum(p.numel() for p in model.encoder.parameters()):,}"
          f"  gate: {sum(p.numel() for p in model.gate_head.parameters()):,}"
          f"  decoder: {sum(p.numel() for p in model.decoder.parameters()):,}")

    #torch.set_float32_matmul_precision("high")

    run_training(model, tr_loader, val_loader, device,
                 steps=STEPS, lr=3e-3, lambda_compress=0.1)

    torch.save({"config": {"d": D, "depth": LAYERS},
                "state_dict": model.state_dict()}, "bcw.pt")
    print("saved → bcw.pt")


if __name__ == "__main__":
    main()
