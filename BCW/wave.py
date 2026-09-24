"https://arxiv.org/abs/2411.02674, https://arxiv.org/abs/2502.02004"


import math
from enum import Enum

import torch
from torch import Tensor
import torch.nn as nn

EPS = 1e-6

def _lift(h: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Lift a real tensor into Wave's (real, imaginary) polar decomposition.

    For each dimension k, computes G_k = ||h_{:,k}||_2 across all tokens,
    then the imaginary component is sqrt(G_k^2 - h_{t,k}^2) — the per-token
    deviation from the global norm.  This is the core Wave representation:
    magnitude encodes global semantics, phase encodes local deviation from it.
    Shared at module level so both WaveLayer and GeometricWaveBlock use the
    same operation without duplication.
    """
    if mask is None:
        h2 = h * h
        G2 = h2.sum(1, keepdim=True)
        return h, (G2.expand_as(h) - h2).clamp(0).add(EPS).sqrt()
    hm  = h * mask
    hm2 = hm * hm
    G2  = hm2.sum(1, keepdim=True)
    return hm, (G2.expand_as(hm) - hm2).clamp(0).add(EPS).sqrt() * mask


class BitFactoredEmbedding(nn.Module):
    """Byte embedding factored over 8 independent bit planes.

    Replaces a 256×dim lookup table with a 2×dim table (one row for bit=0,
    one for bit=1) applied independently to each of the 8 bit planes of the
    byte value.  The byte embedding is the sum of the 8 plane embeddings.

    Parameter count: 2×dim instead of 256×dim — 128x reduction.

    Structural prior: bytes that differ in fewer bits (lower Hamming distance)
    get more similar embeddings by construction.  This is the right inductive
    bias for text and binary data where syntactically related bytes tend to
    share bit patterns — ASCII printable characters cluster in 0x20-0x7E,
    UTF-8 continuation bytes all share the 10xxxxxx pattern, and so on.

    The output head stays as a standard Linear(dim, 256) with cross entropy —
    the independence assumption that breaks bit-factored heads does not apply
    to the input embedding, where the prior is purely about representation
    similarity rather than output probability independence.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.planes = nn.Embedding(2, dim)
        self.register_buffer('bits',
            torch.arange(8, dtype=torch.long).unsqueeze(0))  # (1, 8)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T) byte indices in [0, 255]
        b = (x.unsqueeze(-1) >> self.bits) & 1   # (B, T, 8) bit values
        return self.planes(b).sum(-2)              # (B, T, dim)


class BitFactoredHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.planes = nn.Linear(dim, 8)  # 8 bit predictions

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, dim)
        bits = self.planes(x)              # (B, T, 8)
        # reconstruct logits over 256 bytes from bit predictions
        # or just supervise the 8 bits directly with BCE
        return bits



class WaveLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.l1   = nn.Linear(dim, dim)
        self.l2   = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.mix  = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, mask: Tensor | None = None,
                return_complex: bool = False) -> Tensor:
        m = torch.sigmoid(self.mix)
        r1, i1 = _lift(self.l1(x), mask)
        r2, i2 = _lift(self.l2(x), mask)
        cr = m * (r1*r2 - i1*i2) + (1-m) * (r1+r2)
        ci = m * (r1*i2 + i1*r2) + (1-m) * (i1+i2)
        if mask is not None:
            cr, ci = cr * mask, ci * mask
        if return_complex:
            return torch.view_as_complex(torch.stack([cr, ci], -1))
        mag = (cr*cr + ci*ci).clamp(0).add(EPS).sqrt()
        out = self.norm(mag + x)
        return out * mask if mask is not None else out


class GeometricWaveBlock(nn.Module):
    """A Wave-style block whose second branch is a position rotor rather than
    a second learned linear projection.

    Content branch : l1 → _lift → (r1, i1)
    Position branch: wavelet + SCT → θ → (cos θ, sin θ) = (r2, i2)

    The position branch is a unit-magnitude complex vector by construction —
    the wavelet and SCT produce a phase angle, and (cos θ, sin θ) is the
    corresponding point on the unit circle.  No l2 linear is needed; the
    geometric parameters in WaveletPhase2D and SCTPhase are the only learned
    weights in this branch.

    Under modulation this computes:
        (r1 + i1·j) × (cos θ + sin θ·j) = (r1·cos θ − i1·sin θ)
                                          + (r1·sin θ + i1·cos θ)·j
    which is exactly content rotated by the position angle — a 2D rotor
    acting on content, the Wave analogue of Versor's rotor sandwich product.
    Under interference it is additive superposition.  mix learns the blend.

    Geometric parameters own the conformal generators of Cl(3,1):
        WaveletPhase2D — e1∧e2 (rotation) and e0∧e∞ (dilation)
        SCTPhase       — e1∧e0 and e2∧e0 (special conformal)
    Translations are handled externally via the 2D grid in WaveStack._pos2d.

    Returns complex when return_complex=True (the depth=1 sole-block case).
    Returns real magnitude + residual otherwise, matching WaveLayer's contract
    for stacking.
    """
    def __init__(self, dim: int, num_scales: int):
        super().__init__()
        self.l1      = nn.Linear(dim, dim)
        self.norm    = nn.LayerNorm(dim)
        self.mix     = nn.Parameter(torch.zeros(1))
        self.wavelet = WaveletPhase2D(dim, num_scales)
        self.sct     = SCTPhase()

    def forward(self, x: Tensor, pos2d: Tensor,
                mask: Tensor | None = None,
                return_complex: bool = False) -> Tensor:
        m = torch.sigmoid(self.mix)

        # Content branch — same lift as WaveLayer
        r1, i1 = _lift(self.l1(x), mask)               # (B, T, D)

        # Position branch — unit rotor from conformal phase
        # pos2d is (T, D) after wavelet+SCT; broadcast over batch naturally
        ang = self.wavelet(pos2d) + self.sct(pos2d)     # (T, D)
        r2  = torch.cos(ang)                            # (T, D)
        i2  = torch.sin(ang)                            # (T, D)

        # Wave combination: modulation = rotor action, interference = superposition
        cr = m * (r1*r2 - i1*i2) + (1-m) * (r1+r2)
        ci = m * (r1*i2 + i1*r2) + (1-m) * (i1+i2)
        if mask is not None:
            cr, ci = cr * mask, ci * mask
        if return_complex:
            return torch.view_as_complex(torch.stack([cr, ci], -1))
        mag = (cr*cr + ci*ci).clamp(0).add(EPS).sqrt()
        out = self.norm(mag + x)
        return out * mask if mask is not None else out


class PosMode(str, Enum):
    """Positional encoding mode for WaveStack.

    NONE    — no positional encoding (default).
    SLOTS   — learned per-slot embedding; requires num_slots.
    WAVELET — 2D multi-scale wavelet + SCT; requires width or max_seq.
              String literals ("none", "slots", "wavelet") are accepted
              anywhere a PosMode is expected.
    """
    NONE    = "none"
    SLOTS   = "slots"
    WAVELET = "wavelet"


class WaveStack(nn.Module):
    def __init__(self, dim: int, depth: int,
                 final_complex: bool = True,
                 pos: PosMode | str = PosMode.NONE,
                 num_slots: int | None = None,
                 width: int | None = None,
                 num_scales: int | None = None,
                 max_seq: int | None = None,
                 sparse: bool = False,
                 gate_temp: float = 1.0):
        super().__init__()
        assert depth >= 1
        pos = PosMode(pos)          # accept bare strings transparently
        self.final_complex = final_complex
        self.width  = None          # set below for WAVELET only
        self.geo    = None          # set below for WAVELET only
        self.gate   = None          # set below when sparse=True + WAVELET
        self.phase  = None          # set below for SLOTS only
        # Intermediate Wave layers — depth-1 for WAVELET (geo occupies slot 0),
        # depth-1 for all other modes (self.final occupies the last slot).
        self.layers = nn.ModuleList([WaveLayer(dim) for _ in range(depth - 1)])

        if pos is PosMode.WAVELET:
            assert final_complex, "pos='wavelet' requires final_complex=True"
            # Width: use explicit value, or derive a square grid from max_seq.
            if width is None:
                assert max_seq is not None, \
                    "pos='wavelet' requires either width or max_seq"
                width = math.ceil(math.sqrt(max_seq))
            self.width = width
            # num_scales: use explicit value, or derive from grid dimensions.
            # Each scale halves positional ambiguity, so ceil(log2(N)) scales
            # uniquely code N positions per axis.  Row axis (H) is the binding
            # constraint since it is always >= W for reasonable max_seq values.
            if num_scales is None:
                if max_seq is not None:
                    H          = math.ceil(max_seq / width)
                    num_scales = math.ceil(math.log2(max(H, width, 2)))
                else:
                    num_scales = 4      # explicit width, no seq info → safe default
            self.geo   = GeometricWaveBlock(dim, num_scales)
            self.final = None           # geo is the final block for this path
            if sparse:
                self.gate = SparseGate(dim, gate_temp)

        elif pos is PosMode.SLOTS:
            assert final_complex, "pos='slots' requires final_complex=True"
            assert num_slots is not None, "pos='slots' requires num_slots"
            self.phase = nn.Embedding(num_slots, dim)
            nn.init.zeros_(self.phase.weight)
            self.register_buffer("slot_idx", torch.arange(num_slots))
            self.final = WaveLayer(dim)

        else:   # PosMode.NONE
            self.final = WaveLayer(dim)

    def _pos2d(self, T: int, device: torch.device) -> Tensor:
        """Half-integer normalised (T, 2) grid positions in (-0.5, 0.5).

        Positions are (k + 0.5) / N - 0.5, never landing on +-0.5 exactly.
        This matters because the closed [-1, 1] normalization places the first
        and last tokens at exactly +-1, where sin(pi * 2^s * (+-1)) = 0 for
        every integer scale s — making boundary tokens provably identical to
        the wavelet encoder regardless of how many scales are used.

        The open (-0.5, 0.5) range fixes this, and has a second benefit:
        scale 0 (freq=pi) now spans (-pi/2, pi/2), where sin is strictly
        monotone — so scale 0 alone already assigns every position a unique
        value. Higher scales add finer resolution for adjacent tokens.
        """
        W    = self.width
        H    = math.ceil(T / W)
        idx  = torch.arange(T, device=device)
        rows = ((idx // W).float() + 0.5) / H - 0.5   # (-0.5, 0.5) open
        cols = ((idx  % W).float() + 0.5) / W - 0.5   # (-0.5, 0.5) open
        return torch.stack([rows, cols], dim=-1)        # (T, 2)

    def forward(self, x: Tensor, mask: Tensor | None = None):
        # Gate runs on raw input before any processing.  It replaces the
        # incoming mask with a soft (B, T, D) importance mask — unusual tokens
        # and high-amplitude features get values near 1, others near 0.
        # Padding zeros in the original mask are preserved (hard zeros stay hard).
        if self.gate is not None:
            mask = self.gate(x, mask)

        if self.geo is not None:
            pos2d = self._pos2d(x.shape[1], x.device)
            if not self.layers:                 # depth=1: geo is the sole block
                return self.geo.forward(x, pos2d, mask,
                                        return_complex=self.final_complex)
            # depth>1: geo outputs real, Wave layers follow, last returns complex
            x = self.geo.forward(x, pos2d, mask, return_complex=False)
            for layer in self.layers[:-1]:
                x = layer.forward(x, mask)
            return self.layers[-1].forward(x, mask,
                                           return_complex=self.final_complex)

        # SLOTS / NONE paths — unchanged
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

    def sparsity_cost(self, l1: float = 1.0, l2: float = 0.1) -> Tensor:
        """Elastic net sparsity penalty for the gate's last forward pass.

        Typical training loop usage:
            out  = model(x, mask)
            loss = reconstruction_loss(out, target)
            loss = loss + model.sparsity_cost(l1=current_lambda)
            loss.backward()

        Anneal current_lambda from its initial value toward zero over
        training.  l2 should remain constant throughout to keep the
        JumpReLU thresholds stable.  Once l1 reaches zero the gate is
        guided entirely by the reconstruction loss — the sparsity pattern
        it learned during annealing becomes self-sustaining.
        """
        if self.gate is None:
            raise RuntimeError("sparsity_cost() requires sparse=True")
        return self.gate.sparsity_cost(l1, l2)
