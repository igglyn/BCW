"https://arxiv.org/abs/2411.02674, https://arxiv.org/abs/2502.02004"

import torch
from torch import Tensor
import torch.nn as nn

EPS = 1e-6

class WaveLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.l1   = nn.Linear(dim, dim)
        self.l2   = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.mix  = nn.Parameter(torch.zeros(1))

    def _lift(self, h: Tensor, mask: Tensor|None) -> tuple[Tensor, Tensor]:
        if mask is None:
            h2 = h * h
            G2 = h2.sum(1, keepdim=True)
            return h, (G2.expand_as(h) - h2).clamp(0).add(EPS).sqrt()
        hm  = h * mask
        hm2 = hm * hm
        G2  = hm2.sum(1, keepdim=True)
        return hm, (G2.expand_as(hm) - hm2).clamp(0).add(EPS).sqrt() * mask

    def forward(self, x:Tensor, mask:Tensor|None=None, return_complex:bool=False) -> Tensor:
        m = torch.sigmoid(self.mix)
        r1, i1 = self._lift(self.l1.forward(x), mask)
        r2, i2 = self._lift(self.l2.forward(x), mask)
        cr = m * (r1*r2 - i1*i2) + (1-m) * (r1+r2)
        ci = m * (r1*i2 + i1*r2) + (1-m) * (i1+i2)
        if mask is not None:
            cr, ci = cr * mask, ci * mask
        if return_complex:
            return torch.view_as_complex(torch.stack([cr, ci], -1))
        mag = (cr*cr + ci*ci).clamp(0).add(EPS).sqrt()
        out = self.norm.forward(mag + x)
        return out * mask if mask is not None else out


class WaveStack(nn.Module):
    def __init__(self, dim: int, depth: int,
                 final_complex: bool = True,
                 num_slots: int | None = None):
        super().__init__()
        assert depth >= 1
        self.final_complex = final_complex
        self.layers = nn.ModuleList([WaveLayer(dim) for _ in range(depth - 1)])
        self.final  = WaveLayer(dim)
        self.phase  = None
        if num_slots is not None:
            assert final_complex
            self.phase = nn.Embedding(num_slots, dim)
            _ = nn.init.zeros_(self.phase.weight)
            self.register_buffer("slot_idx", torch.arange(num_slots))

    def forward(self, x: Tensor, mask:Tensor|None=None):
        for layer in self.layers:
            x = layer.forward(x, mask)
        out = self.final.forward(x, mask, return_complex=self.final_complex)
        if self.phase is not None:
            T   = out.shape[1]
            ang = self.phase.forward(self.slot_idx[:T])
            out = out * torch.polar(torch.ones_like(ang), ang).unsqueeze(0)
            if mask is not None:
                out = out * mask
        return out
