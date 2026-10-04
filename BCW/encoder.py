from enum import Enum
import math

import torch
from torch import Tensor
import torch.nn as nn

from BCW.blocks import SparseGate, TilePredictor

def pick_K_ND(context_length: int, n_axes: int, overhead: int = 4) -> int:
    """
    Choose K (number of frequency vectors) for WaveletND.

    context_length : T, number of positions to distinguish
    n_axes         : N, dimensionality of position vectors
    overhead       : practical multiplier over the log2(T) theoretical floor

    K is the larger of:
      K_from_T — information capacity: overhead × ceil(log2(T)), rounded up to
                 a power of two.  Ensures enough sinusoids to separate T positions.
      K_from_N — directional coverage: smallest power of two >= N.  Floor below
                 which the learned vectors cannot span all axes.

    Validated data points (overhead=4):
      T=256    → K=32   (standalone fallback / patch mode)
      T=4096   → K=64   (standard operating point)
      T=131072 → K=128  (full context)
    Crossover N > K_from_T: directional coverage becomes the binding constraint.
    """
    log_T    = math.ceil(math.log2(max(context_length, 2)))
    K_from_T = 1 << math.ceil(math.log2(max(overhead * log_T, 1)))
    K_from_N = 1 << math.ceil(math.log2(max(n_axes, 1)))
    return max(K_from_T, K_from_N)


class WaveletND(nn.Module):
    """
    N-dimensional positional phase encoding with K learnable frequency vectors.

    Replaces the axis-aligned N×S grid of WaveletPhase._ND with K arbitrary
    learned directions in ℝᴺ.  As N grows the vectors adapt — S no longer
    collapses because K is determined by T and N independently via pick_K_ND.

    Structural difference from WaveletPhase._4D
    -------------------------------------------
    Old: K = N × S axis-aligned vectors, one weight set per (axis, scale) pair.
         Capacity is split across axes; S must shrink as N grows at fixed K.
    New: K arbitrary vectors in ℝᴺ; gradient descent finds useful directions.
         Warm-started near axis-aligned (small noise) so the prior is preserved
         while allowing cross-axis combinations to emerge from training.

    Gradient normalisation
    ----------------------
    Divides by sqrt(T) for the same reason as the original modules: the matmul
    backward sums over T tokens into each W row, inflating gradient magnitude
    by ~sqrt(T).  Without this correction, W lurches away from zero on step 1
    exactly as the original wavelet weights would without their own fix.
    """
    def __init__(self, dim: int, n_axes: int, K: int):
        super().__init__()
        # Warm start near axis-aligned: small noise lets gradient descent relax
        # to cross-axis directions without starting from random high-magnitude init
        self.omega = nn.Parameter(torch.randn(K, n_axes) * 0.1)  # (K, N)
        self.W     = nn.Parameter(torch.zeros(K, dim))            # (K, D)

    def forward(self, pos: Tensor) -> Tensor:
        # pos : (T, N) positions normalised to (-0.5, 0.5)
        # returns: (T, D) phase angle per token per dimension
        T   = pos.shape[0]
        phi = torch.sin(pos @ self.omega.T)     # (T, K)
        return (phi @ self.W) / math.sqrt(T)   # (T, D)


class SCTPhaseND(nn.Module):
    """
    Full degree-2 solid harmonic SCT phase correction for N dimensions.

    Parameterises the correction as a traceless symmetric N×N matrix B:
        θ = Σᵢⱼ Bᵢⱼ · pᵢ · pⱼ   where tr(B) = 0

    The existing SCTPhase._4D uses 4 scalar parameters spanning a 4-dimensional
    subspace of the 9-dimensional space of degree-2 solid harmonics for N=4.
    This class spans the full 9-dimensional space (generally (N-1)(N+2)/2 dims).

    The original 4 parameters are recovered when B has only the diagonal and
    the (1,2) off-diagonal entry — existing checkpoints will not warm-start
    cleanly, but zero-init here means the correction starts as a no-op and
    learns in, matching the existing module's behaviour at initialisation.

    Gradient normalisation: divides by sqrt(T) consistent with WaveletND and
    the existing SCTPhase modules (b broadcasts over D, gradient sums over T×D).
    """
    def __init__(self, n_axes: int):
        super().__init__()
        self.n     = n_axes
        self.B_raw = nn.Parameter(torch.zeros(n_axes, n_axes))

    def _B(self) -> Tensor:
        # Symmetrise then enforce traceless via diagonal shift
        B = self.B_raw + self.B_raw.T
        return B - torch.eye(self.n, device=B.device, dtype=B.dtype) * B.diagonal().mean()

    def forward(self, pos: Tensor) -> Tensor:
        # pos : (T, N) — returns (T, 1)
        T     = pos.shape[0]
        theta = torch.einsum("ti,ij,tj->t", pos, self._B(), pos)
        return theta.unsqueeze(-1) / math.sqrt(T)



class TiledEncoder(nn.Module):
    """
    Chunked geometric encoder operating on atomic position tiles.
 
    Replaces GeometricEncoder's full-sequence materialisation with a tile
    loop that processes T positions in chunks of tile_size.  The global
    reductions in _lift (norm² over T) and SparseGate (mean over T) are
    replaced by per-tile local reductions informed by TilePredictor's
    estimate of the accumulated outer state.
 
    Structural differences from GeometricEncoder
    --------------------------------------------
    - tile_size replaces width/depth/height: defines the atomic spatial
      unit directly rather than deriving it from sequence subdivision.
      The position grid falls out from tile_size and n_axes rather than
      being specified as a grid shape.
    - Position._nd replaces _4d/_3d/_2d: generated per tile over the
      tile's position slice, never materialising the full (T, N) grid.
    - WaveletND + SCTPhaseND assumed throughout: no mode routing needed.
    - TilePredictor owns the outer state → predicted ctx/G2 mapping.
    - SparseGate is tile-scoped: deviation computed against predicted
      global mean rather than actual global mean.
    - _lift is tile-scoped: imaginary component computed against
      predicted global norm² rather than actual global norm².
    - Returns (complex output, final outer_state) so the caller can
      thread state across calls for streaming use cases.
 
    Outer state
    -----------
    A (B, 2*D) tensor carrying the running accumulation of ctx and G2
    estimates across tiles.  Updated after each tile via exponential
    moving average so the predictor sees a smoothed history rather than
    raw partial sums.  Initialised to zeros at the start of each forward
    call unless passed explicitly for streaming continuation.
 
    Parameters
    ----------
    dim       : embedding dimension D (4 in current BCW)
    n_axes    : number of spatial axes N (4 for Cl(5,1))
    tile_size : number of positions per tile — the atomic spatial unit
    max_seq   : used to initialise pick_K_ND; not stored after init
    K         : override pick_K_ND result if supplied
    sparse    : whether to apply SparseGate within each tile
    gate_temp : SparseGate temperature
    ema_alpha : smoothing factor for outer state update (0=no update, 1=replace)
    """
    def __init__(self, dim: int,
                 n_axes: int,
                 tile_size: int,
                 max_seq: int | None = None,
                 K: int | None = None,
                 sparse: bool = False,
                 gate_temp: float = 1.0,
                 ema_alpha: float = 0.9):
        super().__init__()
        self.dim       = dim
        self.n_axes    = n_axes
        self.tile_size = tile_size
        self.ema_alpha = ema_alpha
 
        # Axis size per inner dimension — cube root / 4th root of tile_size
        # gives equal-sided tiles; stored as tuple for Position._nd
        d = math.ceil(tile_size ** (1.0 / n_axes))
        # innermost-first convention matching Position._nd
        self.dims = tuple(d for _ in range(n_axes - 1))
 
        # Frequency vector count
        if K is None:
            K = pick_K_ND(max_seq if max_seq is not None else tile_size, n_axes)
        self.K = K
 
        self.l1        = nn.Linear(dim, dim)
        self.wavelet   = WaveletND(dim, n_axes=n_axes, K=K)
        self.sct       = SCTPhaseND(n_axes=n_axes)
        self.predictor = TilePredictor(dim)
        self.gate      = SparseGate(dim, gate_temp) if sparse else None
 
    # ── position grid ──────────────────────────────────────────────────────
 
    def _tile_pos(self, t_start: int, t_end: int,
                  T: int, device: torch.device) -> Tensor:
        """
        Generate position grid for positions [t_start, t_end) within a
        sequence of length T, using the full-sequence axis sizes so that
        tile positions are consistent with what a full _nd call would produce.
 
        Returns (T_tile, N) where T_tile = t_end - t_start.
        """
        T_tile = t_end - t_start
        # Build full-sequence dims: outermost D1 from T, inner from self.dims
        vol = math.prod(self.dims) if self.dims else 1
        D1  = math.ceil(T / vol)
        all_dims = (D1,) + self.dims          # outermost first
 
        idx       = torch.arange(t_start, t_end, device=device)
        remaining = idx.clone()
        coords    = []
        # innermost first
        for d in reversed(all_dims[1:]):
            coords.append((remaining % d).to(torch.float32))
            remaining = remaining // d
        coords.append(remaining.to(torch.float32))      # outermost
        coords.reverse()                      # outermost → innermost
 
        positions = [(c + 0.5) / s - 0.5
                     for c, s in zip(coords, all_dims)]
        return torch.stack(positions, dim=-1)  # (T_tile, N)
 
    # ── phase 1: sequential predictor pass ────────────────────────────────
 
    def _phase1(self, x: Tensor,
                outer_state: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Sequential pass: thread outer state through TilePredictor only.
 
        The only sequential dependency in the architecture is the EMA update
        of outer state — tile N's prediction depends on tile N-1's local
        statistics.  This phase resolves that dependency at minimal cost:
        the predictor is a tiny MLP on a (B, 2*D) tensor, so the Python
        loop overhead is negligible relative to what Phase 2 will do.
 
        No geometric encoding, no gate, no _lift — those are Phase 2.
 
        Parameters
        ----------
        x           : (B, T, D)
        outer_state : (B, 2*D) — initial state, zeros if starting fresh
 
        Returns
        -------
        ctx_preds   : (B, n_tiles, 1, D) — predicted global ctx per tile
        G2_preds    : (B, n_tiles, 1, D) — predicted global G2 per tile
        outer_state : (B, 2*D) — final state for streaming continuation
        """
        B, T, D  = x.shape
        n_tiles  = math.ceil(T / self.tile_size)
        ctx_list = []
        G2_list  = []
 
        for i in range(n_tiles):
            t_start = i * self.tile_size
            t_end   = min(t_start + self.tile_size, T)
 
            # Predict globals from accumulated state before seeing this tile
            ctx_pred, G2_pred = self.predictor(outer_state)  # (B,1,D),(B,1,D)
            ctx_list.append(ctx_pred)
            G2_list.append(G2_pred)
 
            # Update outer state from this tile's local statistics
            x_tile  = x[:, t_start:t_end, :]               # (B, T_tile, D)
            hm      = self.l1(x_tile)                       # (B, T_tile, D)
            ctx_tile = hm.mean(dim=1)                       # (B, D)
            G2_tile  = (hm * hm).sum(dim=1)                # (B, D)
            new_stats = torch.cat([ctx_tile, G2_tile], dim=-1)
            outer_state = (self.ema_alpha * new_stats
                           + (1.0 - self.ema_alpha) * outer_state)
 
        # Stack into (B, n_tiles, 1, D) for Phase 2 indexing
        ctx_preds = torch.stack(ctx_list, dim=1)            # (B, n_tiles, 1, D)
        G2_preds  = torch.stack(G2_list,  dim=1)            # (B, n_tiles, 1, D)
        return ctx_preds, G2_preds, outer_state
 
    # ── phase 2: batched geometric encoding ───────────────────────────────
 
    def _phase2(self, x: Tensor,
                ctx_preds: Tensor,
                G2_preds: Tensor,
                T: int) -> Tensor:
        """
        Batched pass: full geometric encoding over all tiles simultaneously.
 
        Folds the tile dimension into the batch dimension so the GPU sees
        one large operation rather than n_tiles small sequential ones.
        All predicted globals from Phase 1 are already known, so there is
        no sequential dependency here.
 
        Parameters
        ----------
        x         : (B, T_pad, D) — padded to n_tiles * tile_size
        ctx_preds : (B, n_tiles, 1, D)
        G2_preds  : (B, n_tiles, 1, D)
        T         : original unpadded sequence length
 
        Returns
        -------
        output    : (B, T, D) complex
        """
        B        = x.shape[0]
        n_tiles  = ctx_preds.shape[1]
        ts       = self.tile_size
 
        # ── fold tiles into batch ────────────────────────────────────────
        # (B, n_tiles * ts, D) → (B * n_tiles, ts, D)
        x_tiled  = x.reshape(B * n_tiles, ts, D := x.shape[-1])
 
        # predicted globals: (B, n_tiles, 1, D) → (B * n_tiles, 1, D)
        ctx_flat = ctx_preds.reshape(B * n_tiles, 1, D)
        G2_flat  = G2_preds.reshape(B * n_tiles, 1, D)
 
        # ── optional gate: deviation from predicted ctx ──────────────────
        mask = None
        if self.gate is not None:
            dev  = x_tiled - ctx_flat                       # (B*n, ts, D)
            tok  = torch.sigmoid(self.gate.scorer(dev))     # (B*n, ts, 1)
            feat = torch.sigmoid(
                (x_tiled.abs() - self.gate.threshold) / self.gate.beta
            )                                               # (B*n, ts, D)
            mask = tok * feat                               # (B*n, ts, D)
            # Store last tile's mask for sparsity_cost — approximation
            # but consistent with single-tile behaviour
            self.gate._mask = mask
 
        # ── content branch: _lift against predicted G2 ──────────────────
        h   = self.l1(x_tiled)                             # (B*n, ts, D)
        hm  = h * mask if mask is not None else h
        hm2 = hm * hm
        imag = (G2_flat.expand_as(hm) - hm2).clamp(0).add(1e-6).sqrt()
        if mask is not None:
            imag = imag * mask

        #h   = self.l1(x_tiled)
        #hm  = h * mask if mask is not None else h
        # factored form — avoid squared space subtraction
        #G   = (hm * hm / x_tiled.shape[1]).sum(1, keepdim=True).clamp(0).sqrt() * math.sqrt(x_tiled.shape[1])
        #Gh  = G.expand_as(hm)
        #hma = hm.abs()
        #imag = ((Gh - hma) * (Gh + hma)).clamp(0).add(1e-6).sqrt()
        #if mask is not None:
        #    imag = imag * mask

        r1, i1 = hm, imag                                  # (B*n, ts, D)
 
        # ── position branch: full sequence grid, tiled ───────────────────
        # Generate positions for the full padded sequence then reshape
        # so each tile sees its correct absolute positions
        pos_full = self._tile_pos(0, n_tiles * ts, n_tiles * ts,
                                  x.device)                # (n_tiles*ts, N)
        pos_full = pos_full.reshape(n_tiles, ts, self.n_axes)  # (n, ts, N)
        # Expand for batch: (B*n_tiles, ts, N)
        pos_batch = pos_full.unsqueeze(0).expand(
            B, -1, -1, -1
        ).reshape(B * n_tiles, ts, self.n_axes)
 
        # Wavelet and SCT operate on (T_in, N) — flatten tile+batch token dim
        pos_flat = pos_batch.reshape(B * n_tiles * ts, self.n_axes)
        ang_flat = self.wavelet(pos_flat) + self.sct(pos_flat)  # (B*n*ts, D)
        ang      = ang_flat.reshape(B * n_tiles, ts, D)
 
        r2 = torch.cos(ang)                                # (B*n, ts, D)
        i2 = torch.sin(ang)
 
        # ── pure modulation ───────────────────────────────────────────────
        cr = r1*r2 - i1*i2
        ci = r1*i2 + i1*r2
        if mask is not None:
            cr, ci = cr * mask, ci * mask
 
        # ── unfold tiles back to sequence, trim pad ───────────────────────
        cr_seq = cr.reshape(B, n_tiles * ts, D)[:, :T, :]  # (B, T, D)
        ci_seq = ci.reshape(B, n_tiles * ts, D)[:, :T, :]  # (B, T, D)


 
        return cr_seq, ci_seq
                                                           # (B, T, D) complex


 
    # ── full forward ───────────────────────────────────────────────────────
 
    def forward(self, x: Tensor,
                outer_state: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        """
        Two-phase forward pass.
 
        Phase 1 — sequential, cheap:
            TilePredictor threads outer state across tiles to collect
            per-tile predicted ctx and G2.  The only sequential dependency
            in the architecture is resolved here, at the cost of n_tiles
            predictor calls on (B, 2*D) tensors — negligible GPU work.
 
        Phase 2 — parallel, batched:
            Full geometric encoding with all predicted globals known.
            Tiles are folded into the batch dimension; the GPU sees one
            large operation over (B * n_tiles, tile_size, D).
 
        Parameters
        ----------
        x           : (B, T, D)
        outer_state : (B, 2*D) or None — None starts fresh;
                      pass prior state for streaming continuation
 
        Returns
        -------
        output      : (B, T, D) complex
        outer_state : (B, 2*D) for streaming continuation
        """
        B, T, D  = x.shape
        n_tiles  = math.ceil(T / self.tile_size)
        T_pad    = n_tiles * self.tile_size
 
        if outer_state is None:
            outer_state = torch.zeros(B, 2 * D,
                                      device=x.device, dtype=x.dtype)
 
        # Pad to multiple of tile_size for clean reshape in Phase 2
        if T_pad > T:
            x_pad = F.pad(x, (0, 0, 0, T_pad - T))
        else:
            x_pad = x
 
        # Phase 1: sequential predictor pass — resolves EMA dependency
        ctx_preds, G2_preds, outer_state = self._phase1(x, outer_state)
 
        # Phase 2: batched geometric encoding — no sequential dependency
        cr, ci = self._phase2(x_pad, ctx_preds, G2_preds, T)
 
        return cr, ci, outer_state
 
