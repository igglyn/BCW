"""
Byte Conditioning Wave (does need to change def)
================================================
BCW: a byte compressor from a dense byte array to a sparse one
This is just a mostly normal compression pipeline:
 minus not converting to dense;
 dense isn't done yet because it is inconvient to do so

Compression Mechanism:
After encoding, all that is done is per-position gate
 `gated = g * content + (1-g) * pad_embed`
  ratio is the mean of g in all positions, in other words the amount used
This form then can be used to argmax to satasify the decoding process
 - There is no decoder module

Pipeline:
byte_embed -> TiledEncoder -> gate -> gated -> byte_head -> argmax

Losses:
 r1: cross-entropy byte reconstruction (This contains a prediction step)
 ratio: amount of content used

python main.py preprocess <text.txt> <cache_prefix>
python main.py train <cache_prefix> [steps] [batch_size] [workers]
"""

import sys

import torch

from BCW.bcw import BCW
from BCW.dataset import preprocess_bcw, make_loader
from BCW.train import run_training


# ── constants ──────────────────────────────────────────────────────────────
VOCAB   = 256       # bytes 0-255
D_MODEL = 4

N_BYTES = 4096      # fixed input/output sequence length
STEPS = 48000
CHUNK_SIZE = N_BYTES // 16
STRIDE = CHUNK_SIZE   # Left as a debugging option if one needs to test with more samples
BS = 1       # effectivly x2 due to above impl



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
        preprocess_bcw(text_path, out_prefix, N_BYTES * 2)
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

    model = BCW(vocab_size=VOCAB, d=D_MODEL, ctx_length=N_BYTES, chunk_size=CHUNK_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BCW params: {n_params:,}")
    print(f"  encoder: {sum(p.numel() for p in model.encoder.parameters()):,}"
          f"  byte: {sum(p.numel() for p in model.byte_embed.parameters()) + sum(p.numel() for p in model.byte_head.parameters()):,}"
          f"  compress: {sum(p.numel() for p in model.gate_head.parameters()) + sum(p.numel() for p in model.content_proj.parameters()):,}")

    torch.set_float32_matmul_precision("high")

    run_training(model, tr_loader, val_loader, device,
                 steps=STEPS, lr=0.01, stride=STRIDE)

    torch.save({"config": {"d": D_MODEL},
                "state_dict": model.state_dict()}, "bcw.pt")
    print("saved → bcw.pt")


if __name__ == "__main__":
    main()
