"""Trainable control-token projector for EDM control curves."""

from __future__ import annotations

import torch
from torch import nn


class EDMControlConditioner(nn.Module):
    """Compress frame-level EDM controls into extra text-conditioning tokens."""

    def __init__(
        self,
        feature_dim: int,
        text_embed_dim: int = 768,
        token_count: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.text_embed_dim = text_embed_dim
        self.token_count = token_count
        self.pool = nn.AdaptiveAvgPool1d(token_count)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_embed_dim),
        )
        self.token_type = nn.Parameter(torch.zeros(token_count, text_embed_dim))

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        """Return [batch, token_count, text_embed_dim] control tokens."""

        if controls.ndim != 3:
            raise ValueError(f"controls must be [B, T, F], got {tuple(controls.shape)}")
        pooled = self.pool(controls.transpose(1, 2)).transpose(1, 2)
        tokens = self.net(pooled)
        return tokens + self.token_type.unsqueeze(0).to(tokens.dtype)
