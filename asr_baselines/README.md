# asr_baselines

Audio-only ASR finetuned on patient dysarthric speech, as baselines vs the AV
Emformer RNN-T. Standalone from the Lightning module: reuses the patient label
CSVs for `(waveform, text)` but trains via HuggingFace / NeMo.

Goal framing: "do videos help?" These audio models are what the AV model must
beat. Green et al. (Interspeech 2021, Project Euphonia) recipe: freeze most of
the encoder, train only the first ~5 layers, + SpecAugment with **cut frequency
masking, blown-up time masking** (the patient data is tiny + repetitive, so this
is the main overfitting brake).

## Phases

1. **SpecAugment** — `specaugment.py` (Green-parameterized, reusable),
   `configs/specaug_green.yaml`, `visualize_specaug.py` (sanity experiment:
   renders clean-vs-masked log-mel + prints masked fraction). DONE.
2. **Whisper small/medium** — `whisper_finetune.py` / `whisper_eval.py` (HF
   Seq2SeqTrainer; freeze-first-5 encoder layers via `--unfreeze-encoder-layers`;
   built-in SpecAugment, `--no-specaug` for the ablation; selects best on WER).
   SLURM: `slurm/whisper_finetune.sbatch` (submit with SPECAUG=1 and SPECAUG=0
   for the ablation). Deps: `requirements-whisper.txt` (separate env). DONE.
3. **Nemotron streaming** — `make_nemo_manifest.py` (CSV -> NeMo jsonl),
   `nemotron_finetune.py` / `nemotron_eval.py` (FastConformer-CacheAware-RNNT;
   train first 5 encoder layers; NeMo `spec_augment`, `--no-specaug` ablation;
   WER + RTF). SLURM: `slurm/nemotron_finetune.sbatch`. Deps:
   `requirements-nemotron.txt` (its OWN env; NeMo torch pin clashes). DONE.
   Follow-up: cache-aware *streaming* WER+latency (chunked decode w/ carried
   state) vs offline transcribe used now.

## Shared pieces

- `patient_audio.py` — `(waveform, text)` from the AV CSVs; text decoded from
  the canonical SentencePiece token ids, so WER is comparable to the Emformer.
- `metrics.py` — one light normalizer (lowercase, strip punct) + WER/CER for
  ALL models. Uses `jiwer` if present, else a builtin Levenshtein fallback.

## Deps (HPC, not in the local torch-2.6 AV env)

Phase 2 (Whisper): `transformers`, `datasets`, `accelerate`, `jiwer`,
`evaluate`, `librosa`. Phase 3 (Nemotron): `nemo_toolkit[asr]` — pins a
different torch; install in its OWN env, do not mix with the AV env.

## Local sanity (no GPU, no extra deps beyond torch/torchaudio/yaml/matplotlib)

```bash
python -m asr_baselines.visualize_specaug --config asr_baselines/configs/specaug_green.yaml
# on a real clip:
python -m asr_baselines.visualize_specaug --root-dir <ROOT> --label-file <val.csv>
```
