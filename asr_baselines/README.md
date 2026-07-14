# asr_baselines

Audio-only ASR finetuned on patient dysarthric speech, as baselines vs the AV
Emformer RNN-T. Standalone from the Lightning module: reuses the patient label
CSVs for `(waveform, text)` but trains via HuggingFace / NeMo.

Goal framing: "do videos help?" These audio models are what the AV model must
beat. **Current answer: no, not yet** — Whisper (audio-only, large pretrained)
beats every AV Emformer config tested, and even within the AV Emformer itself,
audio-only beats audio-visual. See `results/week_results_2026-06-23_2026-07-01.md`
and `results/week_results_2026-07-02_2026-07-06.md` for the full investigation
and current numbers — this file covers what's in this directory, not the findings.

Green et al. (Interspeech 2021, Project Euphonia) recipe: freeze most of the
encoder, train only the first ~5 layers, + SpecAugment with **cut frequency
masking, blown-up time masking** — tested as the starting point for all three
models. **Finding: it underperforms full finetune on every model size tested**
(Whisper small/medium/turbo/large-v3, Nemotron). Full finetune is now the
recommended default; the freeze recipe is kept in the code as an option, not
the default.

## Phases

1. **SpecAugment** — `specaugment.py` (Green-parameterized, reusable),
   `configs/specaug_green.yaml`, `visualize_specaug.py` (sanity experiment:
   renders clean-vs-masked log-mel + prints masked fraction).
2. **Whisper** — `whisper_finetune.py` / `whisper_eval.py` (HF `Seq2SeqTrainer`;
   freeze-first-5 encoder layers via `--unfreeze-encoder-layers`, or
   `--unfreeze-encoder-layers -1 --train-decoder` for full finetune; built-in
   SpecAugment, `--no-specaug` for the ablation; `load_best_model_at_end` on
   WER). SLURM: `slurm/whisper_finetune.sbatch`. Deps: `requirements-whisper.txt`
   (separate env).

   **Checkpoint-selection caveat**: `load_best_model_at_end` selects the best
   of ~30 epochs using `--val-file` — the same set the final WER used to be
   reported on. That's a real, ~4-6pp optimistic bias (confirmed; it flipped
   which model size looked best before being caught and fixed). `TEST_FILE`
   (sbatch env var, defaults to `VAL_FILE` for back-compat) lets the final
   `whisper_eval.py` call report on a disjoint held-out set instead — use it.
   See `results/week_results_2026-06-23_2026-07-01.md` §28-30 for the full
   story, including the corrected numbers.
3. **Nemotron streaming** — `make_nemo_manifest.py` (CSV → NeMo jsonl),
   `nemotron_finetune.py` / `nemotron_eval.py` (FastConformer-CacheAware-RNNT;
   train first 5 encoder layers or full finetune; NeMo `spec_augment`,
   `--no-specaug` ablation; offline WER + RTF). SLURM:
   `slurm/nemotron_finetune.sbatch`. Deps: `requirements-nemotron.txt` (its
   OWN env; NeMo pins a different torch, do not mix with the Whisper or AV
   envs — see that file for every environment gotcha hit getting this
   pipeline working, including a hard GPU segfault traced to a missing
   `numba-cuda` companion package).

   `nemotron_finetune.py` also has a `SaveBestWER` callback (Nemotron
   originally had **zero** checkpoint selection — always last-epoch) and
   `TEST_FILE` support in the sbatch, same pattern as Whisper. Checked this
   for selection bias the same way as Whisper; **found none** — Nemotron
   never double-dipped, so there was nothing to correct (see
   `results/week_results_2026-07-02_2026-07-06.md` §15).

   `nemotron_streaming_eval.py` / `slurm/nemotron_streaming_eval.sbatch`:
   real cache-aware streaming WER+latency (chunked decode with carried
   state), not just offline `transcribe()` — Nemotron's whole point is
   streaming inference, so the offline number alone was an upper bound, not
   the real deployment number. Streaming WER essentially matched offline WER
   once measured (see `results/week_results_2026-07-02_2026-07-06.md` §12-13).

4. **CarelessWhisper/WhisperRT** (arXiv 2508.12301) — genuinely causal
   streaming Whisper (LoRA + causal attention masks, O(1) per-chunk decode,
   chunks down to 40ms). The route flagged in
   `results/whisper_streaming_investigation_2026-07-09.md` as the only lever
   below SimulStreaming's ~1.4s latency floor.
   `carelesswhisper_streaming_eval.py` (same manifest/metrics/latency contract
   as `whisper_streaming_eval.py`; word-lag is greedy-only — beam decode has
   no streaming timestamps upstream) + `slurm/carelesswhisper_streaming_eval.sbatch`
   (zero-shot released checkpoints; they download anonymously from the public
   HF repo `MLSpeech/CarelessWhisper-Streaming`).
   Patient finetune: `make_carelesswhisper_dataset.py` (MFA corpus + training
   CSV; alignments via `slurm/carelesswhisper_mfa_align.sbatch`) then
   `slurm/carelesswhisper_finetune.sbatch` (papers over upstream quirks:
   `ds_dict_private` import, hardcoded `/mlspeech` output roots). Deps:
   `requirements-carelesswhisper.txt` (its OWN env; pyaudio deliberately
   omitted — the eval script stubs it). Their sizes stop at **large-v2** (no
   v3), and their original code is **CC BY-NC 4.0 (non-commercial)** — flag
   before any clinical/commercial deployment.

## Shared pieces

- `patient_audio.py` — `(waveform, text)` from the AV CSVs; text decoded from
  the canonical SentencePiece token ids, so WER is comparable to the Emformer.
- `metrics.py` — one light normalizer (lowercase, strip punct) + pooled
  corpus WER/CER (`total edits / total ref words`, matching `jiwer`'s
  convention) for ALL models. Uses `jiwer` if present, else a builtin
  Levenshtein fallback. `eval.py` (the AV Emformer's own eval script) was
  found to use a *different* metric (macro-average of per-clip WER) until
  fixed to also report a matching `corpus_wer` — see the main README's
  "Current best results" section and `results/week_results_2026-06-23_2026-07-01.md`
  §27 if comparing older AV numbers against these.

## Methodology: honest three-way splits

Every number in `results/` from mid-investigation onward uses a real
train / val (selection only) / test (final report, never touched by
selection) split, not just train/val. The splits are built once and reused
across all three architectures for consistency — see `results/week_results_2026-06-23_2026-07-01.md`
§29 for exactly how (including a real train/test contamination bug found and
fixed when first attempting a train-corpus × eval-corpus generalization
matrix across legal/legacy/merged domains — reusing independently-carved
per-domain splits to build a merged split leaked ~90% of one domain's "test"
clips into another domain's train set; fixed by deriving the merged split
from the same underlying per-domain train/test partition instead of
re-splitting the merged pool separately).

## Deps (HPC, not in the local torch-2.6 AV env)

Whisper: `transformers`, `datasets`, `accelerate`, `jiwer`, `evaluate`,
`librosa`. Nemotron: `nemo_toolkit[asr]` — pins a different torch; install in
its OWN env, do not mix with the AV or Whisper envs.

## Local sanity (no GPU, no extra deps beyond torch/torchaudio/yaml/matplotlib)

```bash
python -m asr_baselines.visualize_specaug --config asr_baselines/configs/specaug_green.yaml
# on a real clip:
python -m asr_baselines.visualize_specaug --root-dir <ROOT> --label-file <val.csv>
```
