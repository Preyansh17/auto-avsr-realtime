"""SpecAugment for dysarthric-speech ASR finetuning.

Green et al. (Interspeech 2021, Project Euphonia) found that on dysarthric
speech, SpecAugment works best with frequency masking REDUCED and time masking
MASSIVELY INCREASED versus the LibriSpeech defaults. The patient data here is
tiny and repetitive (~300 sentences, 50-word closed vocab, each word >=15
reps), so this aggressive time masking is the main overfitting brake.

Defaults below encode the Green-style direction (cut F, blow up time). Tune via
configs/specaug_green.yaml. This module operates on a log-mel spectrogram of
shape (..., n_mels, time) and is reused by the Whisper and Nemotron pipelines
(both of which also expose equivalent built-in SpecAugment knobs -- use this for
a framework-independent ablation, or map these numbers onto their configs).

Reference defaults for contrast:
  LibriSpeech (SpecAugment paper, "LD"): F=27 x2 freq masks; T=100 x2 time
  masks; p (max masked time fraction) = 1.0.
"""

from dataclasses import dataclass

import torch
import torchaudio.transforms as T


@dataclass
class SpecAugmentConfig:
    # Frequency masking -- REDUCED for dysarthric speech.
    freq_mask_param: int = 13   # max width of a freq mask (mel bins); was 27
    n_freq_masks: int = 1       # number of freq masks; was 2

    # Time masking -- MASSIVELY INCREASED for dysarthric speech.
    time_mask_param: int = 40   # max width of one time mask (frames)
    n_time_masks: int = 10      # number of time masks; was 2
    time_mask_p: float = 0.5    # cap: total masked time <= p * n_frames

    enabled: bool = True

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        known = {k: d[k] for k in d if k in cls.__dataclass_fields__}
        return cls(**known)


class SpecAugment(torch.nn.Module):
    """Apply n_freq_masks frequency masks + n_time_masks time masks (in-place
    safe) to a log-mel spectrogram. No-op in eval mode or when disabled."""

    def __init__(self, config: SpecAugmentConfig = None):
        super().__init__()
        self.config = config or SpecAugmentConfig()
        c = self.config
        # iid_masks=True draws an independent mask per item in a batch.
        self.freq_masks = torch.nn.ModuleList(
            T.FrequencyMasking(freq_mask_param=c.freq_mask_param, iid_masks=True)
            for _ in range(c.n_freq_masks)
        )
        self.time_masks = torch.nn.ModuleList(
            T.TimeMasking(time_mask_param=c.time_mask_param, iid_masks=True, p=c.time_mask_p)
            for _ in range(c.n_time_masks)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (n_mels, time) | (batch, n_mels, time) | (batch, channel,
        n_mels, time). torchaudio's iid masking needs 4D (B, C, F, T); we
        normalize to that, mask, then restore the original rank. Masked cells
        are set to 0 -- pass a mean-normalized mel so 0 sits at the mean."""
        if not self.config.enabled or not self.training:
            return mel
        orig_ndim = mel.dim()
        if orig_ndim == 2:        # (F, T) -> (1, 1, F, T)
            mel = mel[None, None]
        elif orig_ndim == 3:      # (B, F, T) -> (B, 1, F, T)
            mel = mel[:, None]
        elif orig_ndim != 4:
            raise ValueError(f"expected mel of rank 2-4, got {orig_ndim}")
        for m in self.freq_masks:
            mel = m(mel)
        for m in self.time_masks:
            mel = m(mel)
        if orig_ndim == 2:
            return mel[0, 0]
        if orig_ndim == 3:
            return mel[:, 0]
        return mel


def build_specaugment(cfg_dict=None) -> SpecAugment:
    return SpecAugment(SpecAugmentConfig.from_dict(cfg_dict))
