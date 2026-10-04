import time
from math import log


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from BCW.bcw import BCW


def make_stride_batch(patches: torch.Tensor, stride: int) -> torch.Tensor:
    """
    patches : [B, 2W]  — double-context samples from the loader
    stride  : window shift in bytes
    returns : [B * n_windows, W]  where n_windows = W // stride + 1
    """
    B, two_w = patches.shape
    W = two_w // 2
    windows = [patches[:, start:start + W] for start in range(0, W + 1, stride)]
    return torch.cat(windows, dim=0)


@torch.no_grad()
def evaluate(model: BCW, loader: DataLoader, stride: int,
             device: torch.device, n_batches: int = 8) -> dict:
    r1_vals, byte_accs, exact_accs, r1s, = [], [], [], []
    for i, patches in enumerate(loader):
        if i >= n_batches:
            break


        patches = make_stride_batch(patches.to(device, non_blocking=True), stride)
        _, _, losses = model.forward(patches)
        r1, ratio, byte_acc, exact_acc, _ = losses
        r1_vals.append(r1.item())
        r1s.append(ratio.item())
        byte_accs.append(byte_acc)
        exact_accs.append(exact_acc)
        n = max(len(r1_vals), 1)
    return (
        sum(r1_vals)    / n,
        sum(r1s)        / n,
        sum(byte_accs)  / n,
        sum(exact_accs) / n,
        )


@torch.no_grad()
def full_stats(model: BCW, loader: DataLoader, stride: int,
               device: torch.device) -> None:
    correct_b = total_b = correct_c = total_c = 0
    chunk_mses: list[float] = []
    ratios = []

    total = 0

    for patches in loader:
        patches   = make_stride_batch(patches.to(device, non_blocking=True), stride)
        _, _, losses = model.forward(patches)
        _, ratio, byte_acc, exact_acc, _ = losses

        correct_b += byte_acc
        correct_c += exact_acc
        ratios.append(ratio.item())

        total += 1

    print(f"\n── full validation sweep ({total} chunks) ────────────────")
    print(f"byte_acc    {correct_b / total:.6f}"
          f"   ({correct_b*patches.shape[1]:.0f}/{total*patches.shape[0]*patches.shape[1]})")
    print(f"chunk_acc   {correct_c / total:.6f}"
          f"   {correct_c*total:.0f}/{total})")
    print(f"mean_ratio {sum(ratios)/len(ratios):.5f}")

# ── training ───────────────────────────────────────────────────────────────

class LearnedLRScheduler:
    """
    Applies a per-step LR multiplier from the LR head to the optimizer.
    
    Not a torch.optim.lr_scheduler subclass — those assume a predetermined
    schedule. This is a learned schedule that reads from the model's own
    state, so it needs to be called with the multiplier after each forward
    pass rather than stepping on a fixed rule.
    
    base_lr is fixed at construction — the multiplier scales around it
    rather than compounding across steps, so the LR can't drift to
    extremes over long runs.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, warmup: int = 1000):
        self.optimizer = optimizer
        self.base_lr   = base_lr
        self.current_step = 0
        self.warmup    = warmup

    def step(self, multiplier: float) -> None:
        self.current_step += 1
        if self.current_step < self.warmup:
            lr = self.base_lr
        else:
            lr = self.base_lr * float(multiplier)
        for group in self.optimizer.param_groups:
            group['lr'] = lr


    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

def run_training(model: BCW, tr_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 steps:            int   = 4000,
                 stride:           int   = 1,
                 lr:               float = 3e-3) -> None:
    """
    lambda_r1:       cross-entropy weight for pass-1 byte reconstruction.
    lambda_compress: weight on ratio — direct compression budget.
                     Counterbalanced by r1: compress until quality drops.
                     Start at 0.1; increase if model refuses to compress.
    lambda_sparse:   Sarsity loss for weights
    """
    
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    shed = LearnedLRScheduler(opt, lr, warmup=1000)
    #scaler = torch.amp.GradScaler(device)

    t0 = time.time()
    step = 0

    while step < steps:
        for patches in tr_loader:
            if step >= steps:
                break
            patches = make_stride_batch(patches.to(device, non_blocking=True), stride)

            opt.zero_grad(set_to_none=True)
            _, logits, losses = model.forward(patches)
            main, ratio, bacc, eacc, lr_mult = losses

            #ease = lambda val: 1 / (7 + max(3*log(val.detach(), 10), -6))

            loss = (main + ratio)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            opt.step()
            #scaler.scale(loss).backward()
            #scaler.step(opt)
            #scaler.update()
            shed.step(1) #lr_mult)

            if step % 100 == 0 or step == steps - 1:
                val_r1, val_ratio, val_bacc, val_eacc  = evaluate(model, val_loader, stride, device)

                is_zero1 = ratio != 0
                is_one1 = bacc != 1
                is_one2 = eacc != 1
                print(f"step {step:4d}"
                      f"  |  byte={format(bacc, ".4f") if is_one1 else "Exact!"}"
                      f"  exact={format(eacc, ".4f") if is_one2 else "Exact!"}"
                      f"  |  main={main.item():.4f}"
                      f"  ratio={format(ratio.item(), ".4f") if is_zero1 else "WHAT!!"}"
                      #f"  sparse={sparse.item():.4f}"
                      #f"  enc_l1_range={model.encoder.l1.weight.max():.4f}:{model.encoder.l1.weight.min():.4f}"
                      #f"  dec_range={model.decoder.weight.max():.4f}:{model.decoder.weight.min():.4f}"
                      f"  lr={shed.current_lr:.4e}"
                      f"  ][  val_byte={val_bacc:.4f}"
                      f"  val_exact={val_eacc:.4f}"
                      f"  |  val_r1={val_r1:.4f}"
                      f"  val_ratio={val_ratio:.4f}"
                      f"  ({time.time()-t0:.1f}s)")
            step += 1
