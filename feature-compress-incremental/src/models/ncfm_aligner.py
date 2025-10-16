# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn


class NCFMAligner(nn.Module):
    """Lightweight feature aligner used before the incremental classifier.

    The module is trained with characteristic-function matching so that its
    output distribution mimics the one observed in earlier stages.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Residual mapping keeps the aligner close to identity when needed.
        return features + self.net(features)

