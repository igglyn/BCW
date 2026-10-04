import torch
from torch import Tensor
import torch.nn as nn

    
class SparseGate(nn.Module):
    """Soft (B, T, D) importance mask computed from the raw token input.
 
    Two jointly-learned components, both expressed through the existing mask
    parameter — no new pipeline inputs required:
 
    Token gate (inverted ranker):
        Scores each token by deviation from a reference context.  When
        predicted_ctx is supplied (from TiledEncoder's outer state), deviation
        is measured against the predicted global mean — fully relative to what
        the model expects the sequence to look like at this point.  When not
        supplied, falls back to the sequence mean over T (original behaviour).
 
        This is the inverse of Avey's MaxSim criterion: familiar tokens score
        low and are attenuated, unusual tokens score high and are preserved.
        Unusual = reconstruction-critical for compression; familiar =
        predictable from context and therefore compressible.
 
    Feature gate (relative JumpReLU):
        Per-dimension learnable threshold θ_k, now relative to the predicted
        norm rather than an absolute magnitude.  When predicted_G2 is supplied:
            sigmoid((|x_k| / (sqrt(G2_pred_k) + ε) - θ_k) / β)
        θ_k is now a relative threshold — how many predicted-norm units above
        expected a feature must be to pass.  When not supplied, falls back to
        the absolute formulation sigmoid((|x_k| - θ_k) / β).
 
        Self-correction property: if overcompression degrades reconstruction,
        predicted_G2 shifts as the outer state adapts, reframing what counts
        as "significant" without any parameter update.  Features that were
        below threshold against the old reference may pass against the new one.
        This is the mechanism that prevents the unrecoverable overcompression
        seen under pure reconstruction loss with the absolute threshold.
 
        Initialised at zero so all features start at gate ≈ 0.5 regardless
        of whether the relative or absolute formulation is active — no dead
        features at init.
 
    Backward compatibility
    ----------------------
    predicted_ctx and predicted_G2 are optional.  When neither is passed the
    gate reduces exactly to the original absolute formulation — existing
    GeometricEncoder usage is unchanged.  TiledEncoder passes both, activating
    the relative formulation without any call-site changes to SparseGate's
    constructor or sparsity_cost.
 
    The joint soft mask is token_weight × feature_weight, a (B, T, D)
    tensor of values in (0, 1).
 
    Sparsity cost (for training):
        L = λ₁ × mean(mask) + λ₂ × mean(θ²)
        λ₁ drives sparsity via the mask values.
        λ₂ regularises the thresholds — meaning is now relative units above
        predicted norm rather than absolute magnitude, but the regularisation
        purpose (preventing drift to ±∞) is unchanged.
        Anneal λ₁ toward zero over training; hold λ₂ constant.
        The last-computed mask is stored as self._mask for cost computation.
    """
    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        self.scorer    = nn.Linear(dim, 1, bias=False)
        self.threshold = nn.Parameter(torch.zeros(dim))
        self.beta      = temperature
        self._mask: Tensor | None = None
 
    def forward(self, x: Tensor,
                mask: Tensor | None = None,
                predicted_ctx: Tensor | None = None,
                predicted_G2:  Tensor | None = None) -> Tensor:
        # Token gate: deviation from predicted ctx if available, else sequence mean
        ctx = predicted_ctx if predicted_ctx is not None else x.mean(1, keepdim=True)
        dev = x - ctx                                                # (B, T, D)
        tok = torch.sigmoid(self.scorer(dev))                       # (B, T, 1)
 
        # Feature gate: relative to predicted norm if available, else absolute
        if predicted_G2 is not None:
            # Normalise feature magnitude by predicted norm per dimension
            norm_pred = predicted_G2.clamp(0).sqrt() + 1e-6        # (B, 1, D)
            feat = torch.sigmoid(
                (x.abs() / norm_pred - self.threshold) / self.beta
            )                                                        # (B, T, D)
        else:
            # Original absolute formulation — backward compatible
            feat = torch.sigmoid(
                (x.abs() - self.threshold) / self.beta
            )                                                        # (B, T, D)
 
        # Joint soft mask: token importance × feature importance
        soft = tok * feat                                            # (B, T, D)
 
        # Respect incoming padding mask — hard zeros stay hard zeros
        if mask is not None:
            soft = soft * mask
 
        self._mask = soft
        return soft



class TilePredictor(nn.Module):
    """
    Predicts global ctx and G2 from accumulated outer state.
 
    The outer state is a processed summary of all tiles seen so far —
    a fixed-size (B, 2*D) tensor carrying running estimates of the
    sequence mean (ctx) and global norm² (G2).  The predictor refines
    those estimates into the predicted globals that the inner tile uses
    to compute deviations, replacing the full-sequence reductions that
    _lift and SparseGate currently require.
 
    At worst net neutral — a poor predictor approximates a running mean
    and the tile degrades gracefully toward the chunked-mean case.  At
    best it learns to anticipate how ctx and G2 will evolve over the
    remaining tiles, giving the inner mask a better deviation reference
    than any running statistic could provide.
 
    Structurally identical to FIREPos (blocks.py) — an MLP mapping
    bounded input to a learned embedding — applied to state prediction
    rather than positional encoding.  The arrival at the same structure
    through a completely different derivation suggests it is the right
    primitive for this role.
 
    Parameters
    ----------
    dim    : embedding dimension D
    hidden : MLP hidden width; 64 is sufficient given the narrow I/O
    """
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
 
    def forward(self, outer_state: Tensor) -> tuple[Tensor, Tensor]:
        # outer_state : (B, 2*D)
        out      = self.net(outer_state)              # (B, 2*D)
        ctx_pred = out[:, :self.dim].unsqueeze(1)    # (B, 1, D)
        G2_pred  = out[:, self.dim:].unsqueeze(1)    # (B, 1, D)
        return ctx_pred, G2_pred

