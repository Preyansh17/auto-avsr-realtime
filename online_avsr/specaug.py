"""SpecAugment for the AV Emformer, in FUSION-FEATURE space.

The audio frontend here is a raw-waveform Conv1dResNet (no mel spectrogram), so
classic frequency-bin masking has no axis. Instead we mask the fused feature
sequence (B, T, C) the Emformer consumes:
  * TIME masking on the frame axis (T) -- the Green et al. (Interspeech 2021)
    lever for dysarthric speech: "massively increased time masking";
  * CHANNEL masking on the feature axis (C) -- the frequency-masking analog,
    kept light per Green ("reduced frequency masking").

Applied in the training step only (guarded on module.training). Pairs with the
heavier waveform/video time-masking in transforms.py (turned on by the same
--specaug flag); this module adds the channel axis + cross-stream fused masking
those per-stream transforms cannot reach.
"""

import random

import torch
from torch import nn


class FeatureSpecAugment(nn.Module):
    def __init__(
        self,
        n_time_masks: int = 10,
        time_mask_param: int = 12,
        time_mask_p: float = 0.4,
        n_feat_masks: int = 1,
        feat_mask_param: int = 27,
    ):
        super().__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_param = time_mask_param
        self.time_mask_p = time_mask_p
        self.n_feat_masks = n_feat_masks
        self.feat_mask_param = feat_mask_param

    def extra_repr(self) -> str:
        return (f"time(masks={self.n_time_masks}, param={self.time_mask_param}, p={self.time_mask_p}), "
                f"feat(masks={self.n_feat_masks}, param={self.feat_mask_param})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C). No-op in eval mode."""
        if not self.training:
            return x
        b, t, c = x.shape
        x = x.clone()
        for i in range(b):
            budget = int(self.time_mask_p * t)
            for _ in range(self.n_time_masks):
                if budget <= 0:
                    break
                w = random.randint(0, min(self.time_mask_param, budget))
                if w <= 0:
                    continue
                start = random.randint(0, max(0, t - w))
                x[i, start:start + w, :] = 0
                budget -= w
            for _ in range(self.n_feat_masks):
                w = random.randint(0, min(self.feat_mask_param, c))
                if w <= 0:
                    continue
                start = random.randint(0, max(0, c - w))
                x[i, :, start:start + w] = 0
        return x
