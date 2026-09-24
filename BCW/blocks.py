
import torch
from torch import Tensor
import torch.nn as nn



class FIREPos(nn.Module):
    """FIRE-inspired position encoding for the decoder (arxiv 2310.04418).

    Standard nn.Embedding allocates O(max_seq × dim) parameters and hard-walls
    at training length — no valid embedding exists for positions beyond max_seq.
    FIRE replaces the lookup table with a learned function applied to bounded,
    normalised positions, giving length generalisation by construction.

    Normalisation: (t + 0.5) / T ∈ (0, 1) for all T.
    The half-integer offset keeps positions off the boundary (consistent with
    the encoder's _pos2d convention).  The function receives values in (0, 1)
    whether T is 128 or 65536 — the input is always in-distribution.

    Parameter count: O(hidden × dim), constant in sequence length.

    A learned MLP is used rather than a fixed wavelet basis because the decoder's
    position role is qualitatively different from the encoder's.  The encoder
    encodes geometric conformal structure (rotation, dilation, translation, SCT).
    The decoder routes a global Wave latent to the correct reconstruction target
    at each position — an arbitrary learned mapping, not a geometric one.  The
    MLP can represent that mapping freely; a sinusoidal basis would constrain it
    without justification.

    Usage (replacing nn.Embedding in a decoder):
        # init:  self.pos = FIREPos(dim)
        # fwd :  pos = self.pos(T, device)          # (T, dim)
        #        out = self.decoder(x + pos)
    """
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, T: int, device: torch.device) -> Tensor:
        t   = torch.arange(T, device=device).float()
        pos = ((t + 0.5) / T).unsqueeze(-1)    # (T, 1)  bounded in (0, 1)
        return self.net(pos)                     # (T, dim)





class SparseGate(nn.Module):
    """Soft (B, T, D) importance mask computed from the raw token input.

    Two jointly-learned components, both expressed through the existing mask
    parameter — no new pipeline inputs required:

    Token gate (inverted ranker):
        Scores each token by deviation from the global mean across the
        sequence.  This is the inverse of Avey's MaxSim criterion: familiar
        tokens score low and are attenuated, unusual tokens score high and
        are preserved.  Unusual = reconstruction-critical for compression;
        familiar = predictable from context and therefore compressible.
        Implemented as a bias-free linear projection of the per-token mean
        deviation so the model can learn a task-specific definition of
        "unusual" rather than having it hardcoded.

    Feature gate (JumpReLU):
        Per-dimension learnable threshold θ_k.  A sigmoid approximates the
        hard jump: sigmoid((|x_k| - θ_k) / β).  Features exceeding their
        threshold pass through proportionally; others are attenuated.
        Initialised at zero so all features start at gate ≈ 0.5 — no dead
        features at init, sparsity is learned from the loss.

    The joint soft mask is token_weight × feature_weight, a (B, T, D)
    tensor of values in (0, 1).  This replaces the binary padding mask with
    a continuous importance weighting.  Where a binary mask expressed
    "present / absent," the soft mask expresses "how much does this
    position and dimension contribute to the global pool."

    Because the mask enters _lift as hm = h * mask before G is computed,
    tokens with high gate values dominate the global norm.  The inverted
    ranker therefore makes unusual tokens define the global context — the
    opposite of what a similarity-based ranker would do, and exactly right
    for compression where the unusual content carries the most information.

    Sparsity cost (for training):
        L = λ₁ × mean(mask) + λ₂ × mean(θ²)
        λ₁ drives sparsity via the mask values.
        λ₂ regularises the thresholds to prevent drift toward −∞ (all
        features always-on) or +∞ (all features always-off).
        Anneal λ₁ toward zero over training; hold λ₂ constant.
        The last-computed mask is stored as self._mask for cost computation.
    """
    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        self.scorer    = nn.Linear(dim, 1, bias=False)
        self.threshold = nn.Parameter(torch.zeros(dim))
        self.beta      = temperature
        self._mask: Tensor | None = None

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # Token gate: score by deviation from global mean (inverted ranker)
        ctx  = x.mean(1, keepdim=True)                               # (B, 1, D)
        dev  = x - ctx                                               # (B, T, D)
        tok  = torch.sigmoid(self.scorer(dev))                       # (B, T, 1)

        # Feature gate: soft JumpReLU threshold per dimension
        feat = torch.sigmoid((x.abs() - self.threshold) / self.beta) # (B, T, D)

        # Joint soft mask: token importance × feature importance
        soft = tok * feat                                             # (B, T, D)

        # Respect incoming padding mask — hard zeros stay hard zeros
        if mask is not None:
            soft = soft * mask

        self._mask = soft
        return soft

    def sparsity_cost(self, l1: float = 1.0, l2: float = 0.1) -> Tensor:
        """Elastic net penalty on the last forward pass.

        Call after forward() and before optimizer.step().
        Anneal l1 toward zero over training as the reconstruction loss
        takes over; keep l2 constant to hold thresholds stable.
        """
        if self._mask is None:
            raise RuntimeError("sparsity_cost() called before forward()")
        return l1 * self._mask.mean() + l2 * (self.threshold ** 2).mean()
