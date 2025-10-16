# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CharacteristicFunctionLoss(nn.Module):
    """Characteristic function matching loss inspired by NCFM.

    This matches the frequency-domain statistics of two feature batches by
    comparing their amplitude and phase responses over random frequency probes.
    """

    def __init__(
        self,
        num_freqs: int = 128,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_freqs = num_freqs
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        target_features: torch.Tensor,
        student_features: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_features.numel() == 0 or student_features.numel() == 0:
            return torch.tensor(0.0, device=student_features.device, dtype=student_features.dtype)

        feat_dim = student_features.shape[-1]
        if freq is None:
            freq = torch.randn(
                self.num_freqs,
                feat_dim,
                device=student_features.device,
                dtype=student_features.dtype,
            )

        def _cf_stats(feats: torch.Tensor, probes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            proj = feats @ probes.t()
            real = torch.cos(proj).mean(dim=0)
            imag = torch.sin(proj).mean(dim=0)
            return real, imag

        real_t, imag_t = _cf_stats(target_features, freq)
        real_s, imag_s = _cf_stats(student_features, freq)

        amp_t = torch.sqrt(real_t.square() + imag_t.square() + 1e-12)
        amp_s = torch.sqrt(real_s.square() + imag_s.square() + 1e-12)

        amp_diff = (amp_t - amp_s).square()
        phase_term = 2 * (amp_t * amp_s - real_t * real_s - imag_t * imag_s)
        phase_term = phase_term.clamp_min(1e-12)

        loss = torch.sqrt(self.alpha * amp_diff + self.beta * phase_term).mean()
        return loss

