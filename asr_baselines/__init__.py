"""Audio-only ASR baselines finetuned on patient dysarthric speech.

Standalone from the AV Emformer RNN-T: these reuse the patient label CSVs for
(waveform, text) but train via HuggingFace / NeMo, not the Lightning module.

Phases:
  1. SpecAugment (specaugment.py) -- Green-style, reusable across models.
  2. Whisper small/medium finetune (whisper_finetune.py).
  3. Nemotron streaming FastConformer-RNNT finetune (nemotron_finetune.py).
"""

from .specaugment import SpecAugment, SpecAugmentConfig, build_specaugment  # noqa: F401
