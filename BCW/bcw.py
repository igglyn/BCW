"https://arxiv.org/abs/2412.09871"

from typing import Callable, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from BCW.wave import WaveStack


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


def raw_cartesian(z: torch.Tensor) -> torch.Tensor:
    return torch.cat([z.real, z.imag], dim=-1)

class BCW(nn.Module):
    """
    Byte Conditioning Wave.

    Shared components (identical weights used in both passes):
      encoder:      WaveStack — byte-scale phase embedding (num_slots=N_BYTES)
      gate_head:    Linear(2*d, 1) — per-position content/padding decision
      content_proj: Linear(2*d, d) — maps complex encoder output to content d-vector
      decoder:      WaveStack — no phase embedding, takes gated + pos_emb as input
      pos_emb:      Embedding(N_BYTES, d) — positional context for decoder
      pad_embed:    Parameter(d) — learned constant for padding positions

    Separate (not shared):
      byte_embed:   Embedding(VOCAB, d) — byte → embedding (pass 1 input only)
      byte_head:    Linear(d, VOCAB)    — pass 1 byte reconstruction
      lat_head:     Linear(d, d)        — pass 2 latent reconstruction
    """

    def __init__(self, vocab_size: int, ctx_length: int, d: int = 4, depth: int = 1):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size

        # ── pass 1 input embedding ────────────────────────────────────────
        self.byte_embed = nn.Embedding(vocab_size, d)

        # ── shared: compression mechanism ─────────────────────────────────
        # pad_embed: what a "compressed away" position carries.
        # Zero-init so it starts neutral; learns what empty positions mean.
        self.pad_embed    = nn.Parameter(torch.zeros(d))
        self.encoder      = WaveStack(d, depth, final_complex=True,
                                      num_slots=ctx_length)
        self.gate_head    = nn.Linear(2 * d, 1)
        self.content_proj = nn.Linear(2 * d, d)

        # ── shared: decoder ───────────────────────────────────────────────
        # No phase embedding — positional context comes from additive pos_emb.
        self.decoder = WaveStack(d, depth, final_complex=False)
        self.pos_emb = nn.Embedding(ctx_length, d)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # ── task-specific heads (not shared) ──────────────────────────────
        self.byte_head = nn.Linear(d, vocab_size)  # pass 1: byte logits
        self.lat_head  = nn.Linear(d, d)      # pass 2: latent reconstruction

    # ── core shared operations ─────────────────────────────────────────────

    def _encode_gate(self, x: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode x and produce gated output.

        x:       [B, T, d] real
        Returns:
          gated: [B, T, d] real — g*content + (1-g)*pad_embed
          ratio: scalar ∈ (0,1) — mean gate value (compression ratio)

        Gradient path:
          d(loss_compress)/d(gate_head) ≡ 1/N  — always non-zero.
          d(loss_recon)/d(gate_head) proportional to (content - pad_embed) —
          non-zero once the encoder learns to differentiate content from padding.
        """
        B, T = x.shape[0], x.shape[1]

        z      = self.encoder(x)                              # [B, T, d] complex
        z_rc   = raw_cartesian(z)                            # [B, T, 2*d]
        g      = torch.sigmoid(self.gate_head.forward(z_rc))        # [B, T, 1]
        content = self.content_proj.forward(z_rc)                   # [B, T, d]
        pad    = self.pad_embed.view(1, 1, -1).expand(B, T, -1)

        gated  = g * content + (1 - g) * pad               # [B, T, d]
        ratio  = g.mean()
        return gated, ratio

    def _decode(self, gated: torch.Tensor,
                head: nn.Module) -> torch.Tensor:
        """
        Decode gated sequence through the shared decoder and task head.

        gated: [B, T, d] real
        head:  output linear layer
        """
        device  = gated.device
        T       = gated.shape[1]
        pos     = self.pos_emb.forward(torch.arange(T, device=device))
        dec_out = self.decoder(gated + pos.unsqueeze(0))   # [B, T, d]
        return head(dec_out)

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, patches: torch.Tensor) -> tuple[Tensor, Tensor, tuple[ Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """
        patches: [B, N_BYTES] long — raw byte IDs (0-255)

        Pass 1: bytes → encoder → gate → gated1 → decoder → byte_head
          Target: original bytes (cross-entropy)
        Pass 2: gated1 → encoder → gate → gated2 → decoder → lat_head
          Target: gated1 (MSE — reconstruct pass-1 output)
        Both passes use identical shared weights.

        Returns:
          gated1:   [B, N_BYTES, d]    — pass-1 gated output (intermediate)
          loss_r1:  scalar — byte cross-entropy
          loss_r2:  scalar — latent MSE
          ratio1:   scalar — pass-1 compression ratio (lower = more compressed)
          ratio2:   scalar — pass-2 compression ratio


        VICReg is computed outside forward in run_training on gated1 and gated2,
        following the same pattern as the other models.
        """

        PredLoss: TypeAlias = Callable[[Tensor, Tensor], Tensor]
        RepLoss: TypeAlias = Callable[[Tensor], Tensor]

        recon_func: PredLoss = lambda given, _: F.cross_entropy(given.reshape(-1, self.vocab_size), patches.reshape(-1))
        mse_func: PredLoss = lambda given, target: F.mse_loss(given, target.detach())

        stage_loss_func: list[PredLoss] = [recon_func, mse_func]
        stage_heads: list[nn.Linear] = [self.byte_head, self.lat_head]

        universal_loss_func: list[RepLoss] = [PatternLoss.variance, PatternLoss.covariance]

        stage_loss: list[Tensor] = []
        universal_loss: list[Tensor] = []

        ratios: list[Tensor] = []

        # -- pre: embed ----------------------------------------------------
        x = self.byte_embed.forward(patches)                  # [B, 256, d]
        # -- substep: bytes, latents ---------------------------------------

        assert len(stage_loss_func) > 0 and len(stage_heads) > 0
        for idx, val in enumerate(zip(stage_loss_func, stage_heads, strict=True)):
            loss_func, head = val

            gated, ratio = self._encode_gate(x)
            if idx == 0:
                out = gated
            output = self._decode(gated, head)
            if idx == 0:
                giv = output
            stage_loss.append(loss_func(output, x))
            ratios.append(ratio)

            for idx2, uni_loss_func in enumerate(universal_loss_func):
                flat_gated = gated.reshape(-1, self.d)
                res = uni_loss_func(flat_gated)
                if idx == 0:
                    universal_loss.append(res)
                else:
                    universal_loss[idx2] += res

            x = gated

        # -- converstions --------------------------------------------------

        gated1 = out #pyright: ignore[reportPossiblyUnboundVariable]
        logits = giv #pyright: ignore[reportPossiblyUnboundVariable]
        r1, r2 = stage_loss
        var, cov = universal_loss
        ratio1, ratio2 = ratios



        return gated1, logits, (r1, r2, var, cov, ratio1, ratio2)
