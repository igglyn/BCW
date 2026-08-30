
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from BCW.bcw import BCW


class PatternLoss:
    @staticmethod
    def variance(z: torch.Tensor) -> torch.Tensor:
        return F.relu(1.0 - (z.var(dim=0, unbiased=False) + 1e-8).sqrt()).mean()

    @staticmethod
    def covariance(z: torch.Tensor) -> torch.Tensor:
        N, D = z.shape
        z_c  = z - z.mean(0)
        cov  = (z_c.T @ z_c) / max(N - 1, 1)
        sq   = cov.pow(2)
        return (sq.sum() - sq.diagonal().sum()) / D


@torch.no_grad()
def evaluate(model: BCW, loader: DataLoader,
             device: torch.device, n_batches: int = 8) -> dict:
    r1_vals, r2_vals, byte_accs, r1s, r2s = [], [], [], [], []
    for i, patches in enumerate(loader):
        if i >= n_batches:
            break
        patches = patches.to(device, non_blocking=True)
        _, logits, losses = model.forward(patches)
        r1, r2, _, _, ratio1, ratio2 = losses
        r1_vals.append(r1.item())
        r2_vals.append(r2.item())
        r1s.append(ratio1.item())
        r2s.append(ratio2.item())
        byte_accs.append((logits.argmax(-1) == patches).float().mean().item())
    n = max(len(r1_vals), 1)
    return {
        "r1":       sum(r1_vals)   / n,
        "r2":       sum(r2_vals)   / n,
        "byte_acc": sum(byte_accs) / n,
        "ratio1":   sum(r1s)       / n,
        "ratio2":   sum(r2s)       / n,
    }


@torch.no_grad()
def full_stats(model: BCW, loader: DataLoader,
               device: torch.device) -> None:
    correct_b = total_b = correct_c = total_c = 0
    chunk_mses: list[float] = []
    ratio1s, ratio2s = [], []

    total = 0

    for patches in loader:
        patches   = patches.to(device, non_blocking=True)
        gated1, logits, losses = model.forward(patches)
        _, _, _, _, ratio1, ratio2 = losses

        pred       = logits.argmax(-1)
        correct_b += (pred == patches).sum().item()
        total_b   += patches.numel()
        correct_c += (pred == patches).all(dim=1).sum().item()
        total_c   += patches.shape[0]
        ratio1s.append(ratio1.item())
        ratio2s.append(ratio2.item())

        total += len(pred[..., :])

    print(f"\n── full validation sweep ({total} chunks) ────────────────")
    print(f"byte_acc    {correct_b / max(total_b, 1):.6f}"
          f"   ({correct_b}/{total_b})")
    print(f"chunk_acc   {correct_c / max(total_c, 1):.6f}"
          f"   ({correct_c}/{total_c})")
    print(f"mean_ratio1 {sum(ratio1s)/len(ratio1s):.4f}"
          f"   mean_ratio2 {sum(ratio2s)/len(ratio2s):.4f}")

# ── training ───────────────────────────────────────────────────────────────

def run_training(model: BCW, tr_loader: DataLoader,
                 val_loader: DataLoader, device: torch.device,
                 steps:            int   = 4000,
                 lr:               float = 3e-3,
                 lr_min:           float = 1e-4,
                 lambda_r1:        float = 1.0,
                 lambda_r2:        float = 1.0,
                 lambda_compress:  float = 0.1,
                 lambda_var:       float = 0.1,
                 lambda_cov:       float = 0.1) -> None:
    """
    lambda_r1:       cross-entropy weight for pass-1 byte reconstruction.
    lambda_r2:       MSE weight for pass-2 latent reconstruction.
    lambda_compress: weight on (ratio1 + ratio2) — direct compression budget.
                     Counterbalanced by r1 and r2: compress until quality drops.
                     Start at 0.1; increase if model refuses to compress.
    lambda_var/cov:  VICReg on gated sequences [B*N_BYTES, d].
                     Padding positions uniformly carry pad_embed, reducing
                     variance — content positions must compensate, naturally
                     linking compression ratio to representation diversity.
    """
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0,
        total_iters=min(100, steps // 10))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr_min)
    sched  = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[min(100, steps // 10)])

    t0 = time.time()
    step = 0

    while step < steps:
        for patches in tr_loader:
            if step >= steps:
                break
            patches = patches.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            _, logits, losses = model.forward(patches)
            r1, r2, var, cov, ratio1, ratio2 = losses

            loss = (lambda_r1       * r1
                  + lambda_r2       * r2
                  + lambda_compress * (ratio1 + ratio2)
                  + lambda_var      * var
                  + lambda_cov      * cov)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if step % 100 == 0 or step == steps - 1:
                val  = evaluate(model, val_loader, device)
                bacc = (logits.detach().argmax(-1) == patches
                        ).float().mean().item()

                is_zero1 = ratio1 != 0
                is_zero2 = ratio2 != 0
                is_zero3 = cov != 0
                is_one1 = bacc != 1
                print(f"step {step:4d}"
                      f"  |  byte={format(bacc, ".4f") if is_one1 else "Exact!"}"
                      f"  |  r1={r1.item():.4f}"
                      f"  r2={r2.item():.4f}"
                      f"  var={var.item():.4f}  cov={format(cov.item(), ".4f") if is_zero3 else "Zero!"}"
                      f"  ratio1={format(ratio1.item(), ".3f") if is_zero1 else "Zero!"}"
                      f"  ratio2={format(ratio2.item(), ".3f") if is_zero2 else "Zero!"}"
                      f"  lr={sched.get_last_lr()[0]:.2e}"
                      f"  |  val_r1={val['r1']:.4f}"
                      f"  val_byte={val['byte_acc']:.4f}"
                      f"  ({time.time()-t0:.1f}s)")
            step += 1

    full_stats(model, val_loader, device)
