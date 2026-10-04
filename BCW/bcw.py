"https://arxiv.org/abs/2412.09871"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from BCW.encoder import TiledEncoder


class LRHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(2 * dim, 1)
        nn.init.constant_(self.net.bias, 1)

    def forward(self, outer_state: Tensor) -> Tensor:
        raw = self.net(outer_state).mean()
        return 1e-1 * (1e1 ** torch.sigmoid(raw))


class BCW(nn.Module):
    """
    Byte Conditioning Wave.

      encoder:      GeometricEncoder — conformal byte-scale encoder
      gate_head:    Linear(2*d, 1) — per-position content/padding decision
      content_proj: Linear(2*d, d) — maps complex encoder output to content d-vector

      pad_embed:    buffer(d)      — learned constant for compressed-away positions

      byte_embed:   Embedding(VOCAB, d) — byte → embedding
      byte_head:    Linear(d, VOCAB)    — byte reconstruction
    """

    def __init__(self, vocab_size: int, ctx_length: int,
                 d: int = 4, chunk_size: int = 128):
        super().__init__()
        self.d          = d
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size

        self.byte_embed = nn.Embedding(vocab_size, d)

        # pad_embed: what a "compressed away" position carries.
        self.register_buffer("pad_embed", torch.zeros(d))

        self.encoder      = TiledEncoder(d, n_axes=8, max_seq=ctx_length, tile_size=chunk_size, sparse=True)


        self.gate_head    = nn.Linear(2 * d, 1)
        self.content_proj = nn.Linear(2 * d, d)

        # ── task head ─────────────────────────────────────────────────────
        self.byte_head = nn.Linear(d, vocab_size)

        # --- LR head ------------------------------------------------------
        #self.lr_head = LRHead(d)


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


        cr, ci, outer = self.encoder(x)                                    # [B, T, d] complex
        z_rc    = torch.cat([cr, ci], dim=-1)                                   # [B, T, 2*d]
        g       = torch.sigmoid(self.gate_head(z_rc))               # [B, T, 1]
        content = self.content_proj(z_rc)                           # [B, T, d]
        pad     = self.pad_embed.view(1, 1, -1).expand(B, T, -1)

        gated = g * content + (1 - g) * pad                        # [B, T, d]
        ratio = g.mean()
        return gated, ratio, z_rc, outer

    def _chunked_loss(self, gated: Tensor, patches: Tensor
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

            out = self.byte_head(g_chunk)                            # [B, cs, vocab]

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
        tuple[float, float, float, float, float],
    ]:
        """
        patches: [B, N_BYTES] long — raw byte IDs (0-255)

        Returns:
          gated:    [B, N_BYTES, d]    — compressed representation
          output:   [B, N_BYTES] uint8 — predicted bytes (argmax, no grad)
          (r1, ratio byte_acc, exact_acc)
        """
        x = self.byte_embed(patches)                                 # [B, T, d]
        gated, ratio, _, encoder_outer       = self._encode_gate(x)
        #lr_mult = self.lr_head(encode_outer).detach()

        r1, byte_acc, exact_acc, predicted   = self._chunked_loss(gated, patches)

        return gated, None, (
            r1, ratio, byte_acc, exact_acc, 1 #lr_mult
        )
