"""
ESM-CrossDTA — Drug-Target Affinity with Protein Language Model + Cross-Attention.

Reference architecture:
  Öztürk et al. 2018 (DeepDTA) → drug CNN encoder kept.
  ESM-2 (Lin et al. 2023, Science) → frozen protein representation.
  Cross-attention: each drug token attends to every protein residue.

Drug path:
  SMILES tokens → Embedding(64, 128) → 3× Conv1d (same padding) → [B, L_d, d]

Protein path:
  Pre-computed ESM-2 per-residue embeddings [B, L_p, esm_dim]
  → Linear projection → LayerNorm → [B, L_p, d]

Interaction:
  MultiheadAttention(Q=drug, K=V=protein) → drug context [B, L_d, d]
  Residual + LayerNorm → [B, L_d, d]

Pooling:
  max pool drug sequence  → drug_repr  [B, d]
  mean pool protein       → prot_repr  [B, d]

Head:
  concat [B, 2d] → Linear(2d→1024→512→1) → affinity scalar [B]

Attention weights [B, n_heads, L_d, L_p] are exposed for interpretability:
  per-residue importance = attention summed over drug positions and heads.
"""

import torch
import torch.nn as nn


class DrugCNNEncoder(nn.Module):
    """SMILES 1D-CNN that preserves the sequence length (same-padding convolutions)."""

    def __init__(self, vocab_size: int = 64, embed_dim: int = 128,
                 num_filters: int = 96, kernel_size: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        pad = kernel_size // 2
        self.convs = nn.Sequential(
            nn.Conv1d(embed_dim,    num_filters, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv1d(num_filters,  num_filters, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv1d(num_filters,  num_filters, kernel_size, padding=pad), nn.ReLU(),
        )
        self.norm = nn.LayerNorm(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) int tokens → (B, L, num_filters)"""
        x = self.embedding(x).permute(0, 2, 1)   # (B, embed_dim, L)
        x = self.convs(x).permute(0, 2, 1)        # (B, L, num_filters)
        return self.norm(x)


class CrossAttentionDTA(nn.Module):
    def __init__(
        self,
        drug_vocab_size: int = 64,
        embed_dim: int = 128,
        d_model: int = 96,
        esm_dim: int = 320,       # ESM-2 8M (esm2_t6_8M_ur50D) → 320-dim
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_encoder = DrugCNNEncoder(drug_vocab_size, embed_dim, d_model)

        self.prot_proj = nn.Sequential(
            nn.Linear(esm_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Drug tokens (Q) attend to protein residues (K, V).
        # Attention weights reveal which residues drive each drug-token prediction.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),          nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        drug_tokens: torch.Tensor,
        prot_emb: torch.Tensor,
        prot_padding_mask: torch.Tensor = None,
        return_attention: bool = False,
    ):
        """
        drug_tokens:       (B, L_d)           — SMILES integer tokens; 0 = pad
        prot_emb:          (B, L_p, esm_dim)  — per-residue ESM-2 embeddings
        prot_padding_mask: (B, L_p) bool      — True where the position is padding
        return_attention:  if True, also return attn_weights (B, n_heads, L_d, L_p)
        """
        drug_seq  = self.drug_encoder(drug_tokens)         # (B, L_d, d)
        drug_pad  = (drug_tokens == 0)                     # (B, L_d) — padding mask

        prot_seq  = self.prot_proj(prot_emb)               # (B, L_p, d)

        attn_out, attn_weights = self.cross_attn(
            query=drug_seq,
            key=prot_seq,
            value=prot_seq,
            key_padding_mask=prot_padding_mask,
        )
        drug_ctx = self.attn_norm(drug_seq + attn_out)     # (B, L_d, d)

        # Masked max pool: ignore drug padding positions
        masked = drug_ctx.masked_fill(
            drug_pad.unsqueeze(-1).expand_as(drug_ctx), float("-inf")
        )
        drug_repr = masked.max(dim=1).values               # (B, d)

        # Masked mean pool for protein
        if prot_padding_mask is not None:
            real = ~prot_padding_mask                      # True = real residue
            prot_sum  = (prot_seq * real.unsqueeze(-1)).sum(1)
            prot_repr = prot_sum / real.float().sum(1, keepdim=True).clamp(min=1)
        else:
            prot_repr = prot_seq.mean(1)                   # (B, d)

        out = self.fc(torch.cat([drug_repr, prot_repr], dim=1)).squeeze(-1)  # (B,)

        return (out, attn_weights) if return_attention else out
