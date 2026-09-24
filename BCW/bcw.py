"https://arxiv.org/abs/2412.09871"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from BCW.blocks import FIREPos
from BCW.encoder import GeometricEncoder, Position


def raw_cartesian(z: torch.Tensor) -> torch.Tensor:
    return torch.cat([z.real, z.imag], dim=-1)



class Decoder(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()

        self.d = d

        self.linear = nn.Linear(d, d)
        self.norm   = nn.LayerNorm(d)

    def forward(self, chunk: Tensor) -> Tensor:
        return self.norm(self.linear(chunk) + chunk)

class BCW(nn.Module):
    """
    Byte Conditioning Wave.

      encoder:      GeometricEncoder — conformal byte-scale encoder
      gate_head:    Linear(2*d, 1) — per-position content/padding decision
      content_proj: Linear(2*d, d) — maps complex encoder output to content d-vector
      decoder:      Deccoder(d)   — Linear + Norm

      pad_embed:    buffer(d)      — learned constant for compressed-away positions

      byte_embed:   Embedding(VOCAB, d) — byte → embedding
      byte_head:    Linear(d, VOCAB)    — byte reconstruction
    """

    def __init__(self, vocab_size: int, ctx_length: int,
                 d: int = 4, depth: int = 1,
                 chunk_size: int = 4096):
        super().__init__()
        self.d          = d
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

        self.byte_embed = nn.Embedding(vocab_size, d)

        # pad_embed: what a "compressed away" position carries.
        self.register_buffer("pad_embed", torch.zeros(d))

        self.encoder      = GeometricEncoder(d, mode=Position.Mode.FOUR_D,
                                             # Sparsity is not needed, but if this doesn't exist then it doesn't train
                                             max_seq=ctx_length, sparse=True)

        self.gate_head    = nn.Linear(2 * d, 1)
        self.content_proj = nn.Linear(2 * d, d)

        # ── decoder ───────────────────────────────────────────────────────
        self.decoder = Decoder(d)

        # ── task head ─────────────────────────────────────────────────────
        self.byte_head = nn.Linear(d, vocab_size)

    # ── core shared operations ─────────────────────────────────────────────

    def _encode_gate(self, x: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode x and produce gated output.

        x:       [B, T, d] real
        Returns:
          gated: [B, T, d] real — g*content + (1-g)*pad_embed
          ratio: scalar ∈ (0,1) — mean gate value (compression ratio)
          z_rc:  [B, T, 2*d]   — raw-cartesian encoder output (pre-gate)
        """
        B, T = x.shape[0], x.shape[1]

        z       = checkpoint(self.encoder, x, use_reentrant=False)  # [B, T, d] complex
        z_rc    = raw_cartesian(z)                                   # [B, T, 2*d]
        g       = torch.sigmoid(self.gate_head(z_rc))               # [B, T, 1]
        content = self.content_proj(z_rc)                           # [B, T, d]
        pad     = self.pad_embed.view(1, 1, -1).expand(B, T, -1)

        gated = g * content + (1 - g) * pad                        # [B, T, d]
        ratio = g.mean()
        return gated, ratio, z_rc

    def _chunked_decode_loss(self, gated: Tensor, patches: Tensor
                             ) -> tuple[Tensor, float, float, float]:
        """
        Compute r1, byte_acc, exact_acc and predicted bytes

        r1:        mean per-byte cross-entropy.
        byte_acc:  fraction of correctly predicted bytes (plain float, no grad).
        exact_acc: fraction of patches with every byte correct (plain float, no grad).
        predicted: [B, T] uint8 — argmax byte predictions for inference use.
        """
        B, T     = patches.shape
        r1_acc   = []
        pred_acc = []
        correct  = 0

        for start in range(0, T, self.chunk_size):
            end     = min(start + self.chunk_size, T)
            g_chunk = gated[:, start:end]                            # [B, cs, d]
            p_chunk = patches[:, start:end]                          # [B, cs]

            out = self.byte_head(self.decoder(g_chunk))              # [B, cs, vocab]

            r1_acc.append(
                F.cross_entropy(out.reshape(-1, self.vocab_size),
                                p_chunk.reshape(-1), reduction='sum')
            )
            with torch.no_grad():
                pred = out.argmax(-1).to(torch.uint8)              # [B, cs]
                pred_acc.append(pred)
                correct += (pred == p_chunk).sum().item()

        r1          = torch.stack(r1_acc).sum() / (B * T)
        predicted   = torch.cat(pred_acc, dim=1)                    # [B, T] uint8
        byte_acc    = correct / (B * T)
        with torch.no_grad():
            exact_acc = (predicted == patches).all(dim=-1).float().mean().item()

        return r1, byte_acc, exact_acc, predicted

    def forward(self, patches: torch.Tensor) -> tuple[
        Tensor,
        Tensor,
        tuple[float, float, float, float],
    ]:
        """
        patches: [B, N_BYTES] long — raw byte IDs (0-255)

        Returns:
          gated:    [B, N_BYTES, d]    — compressed representation
          output:   [B, N_BYTES] uint8 — predicted bytes (argmax, no grad)
          (r1, ratio byte_acc, exact_acc)
        """
        x = self.byte_embed(patches)                                 # [B, T, d]
        gated, ratio, _                       = self._encode_gate(x)
        r1, byte_acc, exact_acc, predicted   = self._chunked_decode_loss(gated, patches)

        return gated, predicted, (
            r1, ratio, byte_acc, exact_acc,
        )
