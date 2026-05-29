"""
DeepDTA model — PyTorch reproduction.

Reference: Öztürk, Özgür & Ozkirimli (2018), "DeepDTA: deep drug-target
binding affinity prediction", Bioinformatics 34(17):i821-i829. arXiv:1801.10193.

Architecture (from the paper):
  - Two parallel 1D-CNN encoders: one for the SMILES (drug), one for the
    protein sequence. Each is: Embedding -> 3 stacked Conv1d -> global max pool.
  - Concatenate the two learned vectors -> 3 fully-connected layers -> scalar
    affinity (regression). MSE loss.

The paper's encoder widths: filters go 32 -> 64 -> 96, with drug kernel size 4
and protein kernel size 8 (proteins are longer, wider receptive field helps).
FC block: 1024 -> 1024 -> 512 -> 1, dropout 0.1 between FC layers.

"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """One 1D-CNN tower (used for both drug and protein, different sizes)."""

    def __init__(self, vocab_size, embed_dim, num_filters, kernel_size):
        super().__init__()
        # +1 on vocab_size to reserve index 0 for padding
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        # Three stacked conv layers, channels widen: nf -> nf*2 -> nf*3
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len) of integer tokens
        x = self.embedding(x)              # -> (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)             # -> (batch, embed_dim, seq_len) for Conv1d

        # Three stacked 1D convolutions with ReLU. No padding (paper uses 'valid'),
        # so the sequence dimension shrinks by (kernel_size - 1) at each layer.
        # The channels widen because deeper layers compose lower-level motifs
        # (e.g. single SMILES chars -> functional-group-like patterns).
        x = self.relu(self.conv1(x))       # -> (batch, num_filters,     L1)
        x = self.relu(self.conv2(x))       # -> (batch, num_filters*2,   L2)
        x = self.relu(self.conv3(x))       # -> (batch, num_filters*3,   L3)

        # Global max pool over the sequence axis: for each of the num_filters*3
        # feature detectors, keep its strongest activation anywhere in the
        # sequence. This makes the output length-invariant (variable-length
        # SMILES/proteins all map to the same fixed vector) and captures
        # "does this motif appear at all", which is what binding cares about.
        x = torch.max(x, dim=2).values     # -> (batch, num_filters*3)
        return x


class DeepDTA(nn.Module):
    def __init__(
        self,
        drug_vocab_size=64,    # ~64 distinct SMILES chars in the label encoding
        prot_vocab_size=25,    # 20 amino acids + a few ambiguous codes
        embed_dim=128,
        num_filters=32,
        drug_kernel=4,
        prot_kernel=8,
    ):
        super().__init__()
        self.drug_encoder = CNNEncoder(drug_vocab_size, embed_dim, num_filters, drug_kernel)
        self.prot_encoder = CNNEncoder(prot_vocab_size, embed_dim, num_filters, prot_kernel)

        combined_dim = (num_filters * 3) * 2  # two towers concatenated
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, drug_tokens, prot_tokens):
        drug_vec = self.drug_encoder(drug_tokens)   # (batch, num_filters*3)
        prot_vec = self.prot_encoder(prot_tokens)   # (batch, num_filters*3)

        # Concatenate the two learned representations along the feature axis.
        # The FC head then learns the interaction between drug features and
        # protein features that determines binding affinity.
        combined = torch.cat([drug_vec, prot_vec], dim=1)  # (batch, num_filters*3*2)

        out = self.fc(combined)            # (batch, 1)
        return out.squeeze(-1)             # (batch,) to match the label shape
