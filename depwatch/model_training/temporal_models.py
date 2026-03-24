"""Temporal models for abandonment prediction.

Three architectures that consume T=6 monthly feature windows:
- AbandonmentTransformer: 2-layer causal transformer (primary)
- AbandonmentGRU: 1-layer GRU baseline
- AbandonmentMLP: Flattened MLP baseline (no temporal structure)
"""

from __future__ import annotations

import torch
import torch.nn as nn

WINDOW_SIZE = 6
FEATURE_DIM = 24
N_HORIZONS = 3  # 3, 6, 12 month predictions


class AbandonmentTransformer(nn.Module):
    """Sliding-window Transformer for abandonment prediction.

    Input: (batch, T=6, D=24) monthly feature snapshots
    Output: (batch, 3) P(abandoned within 3/6/12 months)
    """

    def __init__(
        self,
        d_input: int = FEATURE_DIM,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, WINDOW_SIZE, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Causal mask: each position can only attend to itself and earlier
        mask = nn.Transformer.generate_square_subsequent_mask(WINDOW_SIZE)
        self.register_buffer("causal_mask", mask)

        # Temporal attention pooling
        self.attn_pool = nn.Linear(d_model, 1)

        # Prediction heads
        self.heads = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, N_HORIZONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, T, D) feature windows

        Returns:
            (batch, 3) logits (sigmoid applied externally)
        """
        # Project and add positional encoding
        h = self.input_proj(x) + self.pos_encoding[:, : x.size(1), :]
        h = self.dropout(h)

        # Transformer encoder with causal mask
        h = self.encoder(h, mask=self.causal_mask)

        # Attention pooling over time dimension
        attn_weights = torch.softmax(self.attn_pool(h), dim=1)  # (batch, T, 1)
        pooled = (h * attn_weights).sum(dim=1)  # (batch, d_model)

        result: torch.Tensor = self.heads(pooled)
        return result


class AbandonmentGRU(nn.Module):
    """GRU baseline for temporal abandonment prediction.

    Purpose-built for short sequences. If this beats the Transformer,
    attention isn't the right inductive bias.
    """

    def __init__(
        self,
        d_input: int = FEATURE_DIM,
        hidden_size: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_input,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, N_HORIZONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)  # h_n: (1, batch, hidden)
        h = self.dropout(h_n.squeeze(0))
        result: torch.Tensor = self.heads(h)
        return result


class AbandonmentMLP(nn.Module):
    """Flattened MLP baseline — no temporal structure.

    Concatenates T=6 snapshots into one vector. Tests whether temporal
    structure matters vs just having more features.
    """

    def __init__(
        self,
        d_input: int = FEATURE_DIM,
        window_size: int = WINDOW_SIZE,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        flat_dim = d_input * window_size
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, N_HORIZONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)  # (batch, T*D)
        result: torch.Tensor = self.net(flat)
        return result
