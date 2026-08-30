
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def preprocess_bcw(text_path: str, out_prefix: str,
                   chunk_size: int,
                   val_frac:   float = 0.05,
                   seed:       int   = 0) -> None:
    """
    Read text as raw UTF-8 bytes, chunk into fixed-size blocks.
    Bytes are naturally 0-255; PAD_ID=256 is reserved for model use.

    Output:
      {out_prefix}.tr.chunks.npy   uint8 [N_tr,  chunk_size]
      {out_prefix}.val.chunks.npy  uint8 [N_val, chunk_size]
    """
    random.seed(seed)
    print(f"reading {text_path} as bytes...")
    raw = open(text_path, "rb").read()
    N   = len(raw) // chunk_size
    print(f"  {len(raw):,} bytes → {N:,} chunks of {chunk_size}")

    chunks = (np.frombuffer(raw[:N * chunk_size], dtype=np.uint8)
              .reshape(N, chunk_size).copy())
    idx    = list(range(N))
    random.shuffle(idx)
    n_val  = max(60, int(N * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    for name, split_idx in [("tr", tr_idx), ("val", val_idx)]:
        arr = chunks[split_idx]
        np.save(f"{out_prefix}.{name}.chunks.npy", arr)
        print(f"  {name}: {len(split_idx):,} → {out_prefix}.{name}.chunks.npy")
    print("done.")


class ChunkDataset(Dataset):
    def __init__(self, path: str):
        self.data = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.data[idx], dtype=np.int64))


def make_loader(out_prefix: str, split: str, bs: int,
                shuffle: bool = True,
                num_workers: int = 4) -> DataLoader:
    ds = ChunkDataset(f"{out_prefix}.{split}.chunks.npy")
    print(f"  {split}: {len(ds):,} chunks")
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
