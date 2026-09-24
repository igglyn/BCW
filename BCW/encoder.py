
from enum import Enum
import math

import torch
from torch import Tensor
import torch.nn as nn

from BCW.wave import _lift
from BCW.blocks import SparseGate

# pyright: reportPrivateUsage=false

class Position:
    class Mode(str, Enum):
        """Geometry mode for GeometricEncoder.

        TWO_D   — 2D conformal grid (default).  Uses WaveletPhase2D + SCTPhase.
                  Auto-square: width = ceil(sqrt(max_seq)).
        THREE_D — 3D conformal grid.  Uses WaveletPhase3D + SCTPhase3D.
                  Auto-cube: width = depth = ceil(max_seq^(1/3)).
        FOUR_D  — 4D conformal grid (Cl(5,1)).  Uses WaveletPhase4D + SCTPhase4D.
                  Auto-hypercube: width = depth = height = ceil(max_seq^(1/4)).
                  Primary target for image/video where spatial + channel axes
                  tile naturally into a balanced 4D grid.

        The additional rotation planes beyond e1e2 are approximated in all
        modes — see WaveletPhase3D/4D docstrings.  String literals accepted.
        """
        TWO_D   = "2d"
        THREE_D = "3d"
        FOUR_D  = "4d"


    @staticmethod
    def _2d(T: int, width: int, device: torch.device) -> Tensor:
        """Half-integer normalised (T, 2) grid positions in (-0.5, 0.5).

        Module-level so both WaveStack and GeometricEncoder can use it without
        either owning it.  width is passed explicitly rather than read from self.
        """
        H    = math.ceil(T / width)
        idx  = torch.arange(T, device=device)
        rows = ((idx // width).float() + 0.5) / H     - 0.5
        cols = ((idx  % width).float() + 0.5) / width - 0.5
        return torch.stack([rows, cols], dim=-1)        # (T, 2)

    @staticmethod
    def _3d(T: int, d2: int, d3: int, device: torch.device) -> Tensor:
        """Half-integer normalised (T, 3) grid positions in (-0.5, 0.5).

        The outermost axis d1 = ceil(T / (d2 * d3)) is computed from T at
        forward time, analogous to how _pos2d computes H from T and width.
        d2 and d3 are the two inner face dimensions (stored at init).
        For the auto-cube case d2 = d3 = ceil(max_seq^(1/3)).
        """
        face = d2 * d3
        D1   = math.ceil(T / face)
        idx  = torch.arange(T, device=device)
        a1   = idx // face
        a2   = (idx % face) // d3
        a3   = idx % d3
        p1   = ((a1.float() + 0.5) / D1) - 0.5
        p2   = ((a2.float() + 0.5) / d2) - 0.5
        p3   = ((a3.float() + 0.5) / d3) - 0.5
        return torch.stack([p1, p2, p3], dim=-1)        # (T, 3)

    @staticmethod
    def _4d(T: int, d2: int, d3: int, d4: int, device: torch.device) -> Tensor:
        """Half-integer normalised (T, 4) grid positions in (-0.5, 0.5).

        The outermost axis d1 = ceil(T / (d2*d3*d4)) is computed at forward
        time.  d2, d3, d4 are the three inner face dimensions (stored at init).
        For the auto-hypercube: d2 = d3 = d4 = ceil(max_seq^(1/4)).
        """
        vol  = d2 * d3 * d4
        D1   = math.ceil(T / vol)
        idx  = torch.arange(T, device=device)
        a4   = idx % d4
        a3   = (idx // d4) % d3
        a2   = (idx // (d4 * d3)) % d2
        a1   = idx // (d4 * d3 * d2)
        p1   = ((a1.float() + 0.5) / D1) - 0.5
        p2   = ((a2.float() + 0.5) / d2) - 0.5
        p3   = ((a3.float() + 0.5) / d3) - 0.5
        p4   = ((a4.float() + 0.5) / d4) - 0.5
        return torch.stack([p1, p2, p3, p4], dim=-1)    # (T, 4)



class WaveletPhase:
    class _2D(nn.Module):
        """
        Multi-scale 2D positional phase encoding.

        Covers the e1e2 (rotation) and e0e_inf (dilation) generators of Cl(3,1).
        RoPE is a single-scale special case of this; using multiple scales gives
        the wavelet-like multi-resolution property from arxiv 2502.02004.

        Weights initialised to zero so the encoding starts as a no-op and is
        learned in from scratch, matching the original phase embedding behaviour.
        """
        def __init__(self, dim: int, num_scales: int = 4):
            super().__init__()
            # (axis=2, scale=S, dim=D) — one weight set per spatial axis
            self.weights = nn.Parameter(torch.zeros(2, num_scales, dim))
            # Fixed scale progression: π, 2π, 4π, 8π, … — explicit float32 to
            # avoid float64 promotion from int64 arange via Python scalar pow.
            self.register_buffer("scales",
                             math.pi * (2.0 ** torch.arange(num_scales, dtype=torch.float32)))

        def forward(self, pos2d: Tensor) -> Tensor:
            # pos2d : (T, 2)  positions normalised to [-1, 1]
            # returns: (T, D)  phase angle per token per dimension
            T      = pos2d.shape[0]
            scaled = pos2d.unsqueeze(-1) * self.scales          # (T, 2, S)
            feats  = torch.sin(scaled)                          # (T, 2, S)
            # Divide by sqrt(T): the einsum backward sums gradients over all T
            # tokens into each weight, inflating gradient magnitude by ~sqrt(T).
            # The previous fix divided by sqrt(2*S) ≈ 2.8 — wrong dimension.
            # For T=512 the correct factor is ~22.6; without it the weights lurch
            # away from zero on step 1 and impose large rotations that corrupt
            # the Wave layers' gradient signal on all subsequent steps.
            return torch.einsum("tas,asd->td", feats, self.weights) / math.sqrt(T)

    class _3D(nn.Module):
        """
        Multi-scale 3D positional phase encoding.

        Extends WaveletPhase2D to three spatial axes, covering:
            e1e∞, e2e∞, e3e∞  — all three translations (via 3D grid positions)
            e0e∞               — dilation (via multi-scale frequencies)
            e1e2, e2e3, e3e1   — rotations approximated (see note below)

        Note on rotations: the complex number constraint means only one rotation
        plane (e1e2) can be represented exactly as a rotor. The additional planes
        e2e3 and e3e1 are approximated through learned linear combinations of the
        three-axis wavelet features. This is the same pragmatic trade-off made
        when dilation was empirically found unnecessary — the model learns what
        it needs within the available structure.

        Weights (3, S, D) versus (2, S, D) for 2D; same einsum, third axis added.
        """
        def __init__(self, dim: int, num_scales: int = 4):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(3, num_scales, dim))
            self.register_buffer("scales",
                             math.pi * (2.0 ** torch.arange(num_scales, dtype=torch.float32)))

        def forward(self, pos3d: Tensor) -> Tensor:
            # pos3d : (T, 3)  positions normalised to (-0.5, 0.5)
            # returns: (T, D)
            T      = pos3d.shape[0]
            scaled = pos3d.unsqueeze(-1) * self.scales   # (T, 3, S)
            feats  = torch.sin(scaled)                   # (T, 3, S)
            return torch.einsum("tas,asd->td", feats, self.weights) / math.sqrt(T)

    class _4D(nn.Module):
        """
        Multi-scale 4D positional phase encoding for Cl(5,1) geometry.

        Extends to four spatial axes, covering all four translation generators
        e1e∞–e4e∞ and dilation e0e∞.  Rotation planes (C(4,2)=6 of them) are
        approximated through learned combinations of four-axis features, same
        pragmatic trade-off as in 3D.

        Weights (4, S, D); same einsum structure throughout all dimensions.
        """
        def __init__(self, dim: int, num_scales: int = 4):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(4, num_scales, dim))
            self.register_buffer("scales",
                             math.pi * (2.0 ** torch.arange(num_scales, dtype=torch.float32)))

        def forward(self, pos4d: Tensor) -> Tensor:
            # pos4d : (T, 4)  positions normalised to (-0.5, 0.5)
            # returns: (T, D)
            T      = pos4d.shape[0]
            scaled = pos4d.unsqueeze(-1) * self.scales   # (T, 4, S)
            feats  = torch.sin(scaled)                   # (T, 4, S)
            return torch.einsum("tas,asd->td", feats, self.weights) / math.sqrt(T)


class SCTPhase:
    class _2D(nn.Module):
        """
        Special conformal transformation phase correction.

        Covers the e1e0 and e2e0 generators of Cl(3,1) — the two generators
        that remain after rotation (wavelet), dilation (wavelet multi-scale),
        and translation (2D grid) are accounted for.

        The phase is the real part of b * p² in complex notation:
            θ(p1, p2) = b1*(p1²-p2²) - 2*b2*(p1*p2)

        Two scalar parameters total, broadcast uniformly across embedding
        dimensions — keeping the generator count honest.
        """
        def __init__(self):
            super().__init__()
            self.b = nn.Parameter(torch.zeros(2))

        def forward(self, pos2d: Tensor) -> Tensor:
            # pos2d : (T, 2)  returns (T, 1)  broadcast over dim in the caller
            T      = pos2d.shape[0]
            p1, p2 = pos2d[:, 0], pos2d[:, 1]
            theta  = self.b[0] * (p1*p1 - p2*p2) - self.b[1] * (2.0 * p1 * p2)
            # b broadcasts over D dimensions, so its gradient sums over both T
            # and D — normalise by sqrt(T) to keep step sizes comparable to the
            # wavelet weights after their own sqrt(T) correction.
            return theta.unsqueeze(-1) / math.sqrt(T)

    class _3D(nn.Module):
        """
        Special conformal transformation phase correction for 3D.

        Covers e1e0, e2e0, and e3e0 — all three SCT generators of Cl(4,1).

        The 2D formula was Re(b * p²) in complex notation, giving:
            b1*(p1²-p2²) - 2*b2*(p1*p2)

        The 3D extension adds the third generator via the l=2, m=0 solid
        harmonic (3*p3² - |p|²) ∝ (2*p3² - p1² - p2²):
            θ = b1*(p1²-p2²) - 2*b2*(p1*p2) + b3*(2*p3²-p1²-p2²)

        Reduces to the 2D formula when p3=0 and b3=0. The third term is
        orthogonal to the first two — it captures the deviation of the third
        axis from the plane defined by the first two.
        Three scalar parameters, broadcast over embedding dimensions.
        """
        def __init__(self):
            super().__init__()
            self.b = nn.Parameter(torch.zeros(3))

        def forward(self, pos3d: Tensor) -> Tensor:
            # pos3d : (T, 3)  returns (T, 1)
            T           = pos3d.shape[0]
            p1, p2, p3  = pos3d[:, 0], pos3d[:, 1], pos3d[:, 2]
            theta = (self.b[0] * (p1*p1 - p2*p2)
                  - self.b[1] * (2.0 * p1 * p2)
                  + self.b[2] * (2.0*p3*p3 - p1*p1 - p2*p2))
            return theta.unsqueeze(-1) / math.sqrt(T)

    class _4D(nn.Module):
        """
        Special conformal transformation phase correction for 4D (Cl(5,1)).

        Covers all four SCT generators e1e0, e2e0, e3e0, e4e0.

        Extends the 3D formula by adding the fourth axis via the next solid
        harmonic term (2*p4² - p1² - p2² - p3²):
            θ = b1*(p1²-p2²) - 2*b2*(p1*p2)
            + b3*(2*p3²-p1²-p2²)
            + b4*(2*p4²-p1²-p2²-p3²)

        The four terms are numerically orthogonal on the 4D unit sphere
        (verified: all pairwise inner products < 0.002) and linearly
        independent. Reduces to the 3D formula when p4=0 and b4=0.
        Four scalar parameters, broadcast over embedding dimensions.
        """
        def __init__(self):
            super().__init__()
            self.b = nn.Parameter(torch.zeros(4))

        def forward(self, pos4d: Tensor) -> Tensor:
            # pos4d : (T, 4)  returns (T, 1)
            T                = pos4d.shape[0]
            p1, p2, p3, p4   = pos4d[:, 0], pos4d[:, 1], pos4d[:, 2], pos4d[:, 3]
            theta = (self.b[0] * (p1*p1 - p2*p2)
                  - self.b[1] * (2.0 * p1 * p2)
                  + self.b[2] * (2.0*p3*p3 - p1*p1 - p2*p2)
                  + self.b[3] * (2.0*p4*p4 - p1*p1 - p2*p2 - p3*p3))
            return theta.unsqueeze(-1) / math.sqrt(T)

class GeometricEncoder(nn.Module):
    """Pure modulation encoder for the compression pipeline.

    Empirically the encoder always converges to the modulation regime under
    compression pressure — interference is a transitional phase that at large
    sequence lengths (131k+) takes hundreds of thousands of steps to exit
    naturally.  Starting in modulation skips that detour entirely.

    Division of labour with a pure-interference decoder:
        Encoder (this): rotates content by position — modulation
        Decoder:        adds position to latent — interference
    Two complementary operations, no parameters wasted on learning which
    regime to use.

    Content branch : l1 → _lift → (r1, i1)
    Position branch: wavelet + SCT → θ → (cos θ, sin θ) = (r2, i2)
    Output         : complex  (r1·r2 − i1·i2) + i(r1·i2 + i1·r2)

    No mix parameter, no interference path, no norm, no residual, no depth
    handling.  Returns complex directly — always the sole encoder block.

    Grid dimensions:
        2D: width (inner), H computed from T at forward time.
            Auto-square: width = ceil(sqrt(max_seq)).
        3D: width, depth (inner two), D1 computed from T.
            Auto-cube: width = depth = ceil(max_seq^(1/3)).
        4D: width, depth, height (inner three), D1 computed from T.
            Auto-hypercube: width = depth = height = ceil(max_seq^(1/4)).

    Optional SparseGate is applied to the raw input before modulation.
    If used, call encoder.gate.sparsity_cost() in the training loop.
    """
    def __init__(self, dim: int,
                 mode: Position.Mode | str = Position.Mode.TWO_D,
                 width: int | None = None,
                 depth: int | None = None,
                 height: int | None = None,
                 max_seq: int | None = None,
                 num_scales: int | None = None,
                 sparse: bool = False,
                 gate_temp: float = 1.0):
        super().__init__()
        mode = Position.Mode(mode)
        self.mode = mode
        self.l1   = nn.Linear(dim, dim)

        if mode is Position.Mode.FOUR_D:
            if width is None:
                assert max_seq is not None, "FOUR_D requires width or max_seq"
                width = math.ceil(max_seq ** (1.0 / 4.0))
            depth  = depth  or width
            height = height or width
            self.width = width; self.depth = depth; self.height = height
            if num_scales is None:
                if max_seq is not None:
                    vol        = width * depth * height
                    D1         = math.ceil(max_seq / vol)
                    num_scales = math.ceil(math.log2(max(D1, width, depth, height, 2)))
                else:
                    num_scales = 4
            self.wavelet = WaveletPhase._4D(dim, num_scales)
            self.sct     = SCTPhase._4D()

        elif mode is Position.Mode.THREE_D:
            if width is None:
                assert max_seq is not None, "THREE_D requires width or max_seq"
                width = math.ceil(max_seq ** (1.0 / 3.0))
            depth  = depth or width
            self.width = width; self.depth = depth; self.height = None
            if num_scales is None:
                if max_seq is not None:
                    face       = width * depth
                    D1         = math.ceil(max_seq / face)
                    num_scales = math.ceil(math.log2(max(D1, width, depth, 2)))
                else:
                    num_scales = 4
            self.wavelet = WaveletPhase._3D(dim, num_scales)
            self.sct     = SCTPhase._3D()

        else:  # TWO_D
            if width is None:
                assert max_seq is not None, "TWO_D requires width or max_seq"
                width = math.ceil(math.sqrt(max_seq))
            self.width = width; self.depth = None; self.height = None
            if num_scales is None:
                if max_seq is not None:
                    H          = math.ceil(max_seq / width)
                    num_scales = math.ceil(math.log2(max(H, width, 2)))
                else:
                    num_scales = 4
            self.wavelet = WaveletPhase._2D(dim, num_scales)
            self.sct     = SCTPhase._2D()

        self.gate = SparseGate(dim, gate_temp) if sparse else None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.gate is not None:
            mask = self.gate(x, mask)

        T = x.shape[1]
        if self.mode is Position.Mode.FOUR_D:
            pos = Position._4d(T, self.height, self.depth, self.width, x.device)
        elif self.mode is Position.Mode.THREE_D:
            pos = Position._3d(T, self.depth, self.width, x.device)
        else:
            pos = Position._2d(T, self.width, x.device)

        # Content branch
        r1, i1 = _lift(self.l1(x), mask)                   # (B, T, D)

        # Position branch — unit rotor, no linear projection needed
        ang = self.wavelet(pos) + self.sct(pos)             # (T, D)
        r2  = torch.cos(ang)
        i2  = torch.sin(ang)

        # Pure modulation: content rotated by position angle
        cr = r1*r2 - i1*i2
        ci = r1*i2 + i1*r2
        if mask is not None:
            cr, ci = cr * mask, ci * mask

        return torch.view_as_complex(torch.stack([cr, ci], -1))
