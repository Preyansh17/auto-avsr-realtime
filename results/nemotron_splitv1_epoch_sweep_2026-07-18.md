# Nemotron on the audited split_v1: epoch sweep, cross-domain generalization, best config

**Date:** 2026-07-18 → 2026-07-19
**Branch:** `realtime-port`

## Why this file exists

Every prior Nemotron result in this repo (`results/week_results_2026-07-14_2026-07-17.md` and earlier) was trained/evaluated on ad-hoc per-domain splits built independently for Nemotron, not on `patient_legal298_legacy96_split_v1` — the newer, independently-audited three-domain split (`split_audit.json`, seed=7) that Whisper's own leaderboard numbers in `README.md` are built on. Nemotron had never been run on split_v1 at all until this investigation. This file documents that gap being closed, plus the epoch-count sweep that followed from it.

## Setup

Split_v1's label CSVs (`labels/{legal,legacy,merged}/{train,val,test}.csv`) carry unigram5000 token ids, not Nemotron's 1023-piece vocab — re-tokenized all 9 via `scripts/regenerate_patient_labels.py` into `/scratch/pa2753/avsr_realtime/labels/splitv1/`. Row counts match `split_audit.json` exactly (legal 238/30/30, legacy 76/10/10, merged 314/40/40). `ROOT_DIR` for manifest-building is `patient_legal298_legacy96_split_v1` directly — it already carries the same per-source symlinked subdirs (`patient_retinaface_legal298`, `patient_retinaface_legacy`) that `slurm/_patient_data.sh`'s own merge logic builds, so audio paths resolve identically regardless of which merged root a job ends up pointed at (see caveat below).

**Caveat on `_patient_data.sh`:** the sbatch's `merged` branch unconditionally overwrites `ROOT_DIR` back to its own `MERGED_ROOT` default even when a caller passes `ROOT_DIR=` via `--export` (unlike `TRAIN_FILE`/`VAL_FILE`/`TEST_FILE`, which the script does respect as overrides). This didn't break anything here only because `MERGED_ROOT`'s own symlinks happen to point at the exact same source directories as split_v1's — confirmed by inspection before trusting the first training run's manifests. Worth fixing properly (`ROOT_DIR="${ROOT_DIR:-$MERGED_ROOT}"`) if `ROOT_DIR` override is ever needed for a genuinely different data root.

## Approach: train once on merged, evaluate everywhere

Following the pattern established on the ad-hoc split (`results/week_results_2026-07-14_2026-07-17.md` §2): train a single full-FT model on split_v1's `merged` domain, then cross-domain-evaluate the same checkpoint on `legal` and `legacy`'s own split_v1 test sets, rather than training three separate domain-specific models.

## Epoch sweep (merged, in-domain, offline greedy WER, 3 seeds each)

| Epochs | Seed1 | Seed2 | Seed3 | Median |
|---|---|---|---|---|
| 90 | 20.63% | 21.08% | 20.63% | 20.63% |
| 120 | 10.76% | 16.59% | 14.80% | 14.80% |
| 150 | 13.00% | 13.00% | 13.90% | 13.00% |
| **180** | 10.76% | 12.11% | 13.90% | **12.11%** |

Still improving at 180, no plateau found (120→150: −1.8pp, 150→180: −0.9pp, diminishing but real and monotonic in the median). **This directly contradicts the epoch-reversal seen on the ad-hoc split**, where 120 epochs (single-seed) reversed against 90's 3-seed median (33.3%→18.0%→~19.1%). The epoch-count sweet spot is split-dependent, not a fixed property of the model/recipe — a genuinely unexpected finding. Not pushed past 180 (diminishing returns vs. ~2-2.5h/seed compute cost); a real ceiling has not been established.

Val-WER trajectory for the epochs=180 seed1 run (10.76% final test WER) was still dropping meaningfully at its best checkpoint (epoch 107 of 180, val_wer 21.0%, down from 22.3% at epoch 67) — the selected checkpoint is not from deep into an overfitting plateau.

## Full cross-domain / streaming / beam battery — epochs=120 (most completely characterized checkpoint)

3 seeds, all three domains, offline greedy, offline+beam (`maes`, width 8), and real cache-aware streaming decode with word-level latency (`nemotron_streaming_eval.py --word-lag`).

| Domain | Offline greedy (median) | Offline +beam (median) | **Streaming, deployable (median)** | TTFT-from-speech | Word-commit lag |
|---|---|---|---|---|---|
| merged (in-domain) | 14.80% | 13.00% | **13.90%** | ~0.95s | ~0.88s |
| legal (cross-domain) | 13.11% | 10.38% | **12.02%** | ~0.97s | ~0.87s |
| legacy (cross-domain) | 20.00% | 25.00%¹ | **20.00%** | ~0.91s | ~0.87s |

¹ Beam decode *hurts* legacy specifically — the one exception to beam helping everywhere else this whole investigation. N=10, plausibly noise, not investigated further.

Streaming WER equals offline greedy WER exactly on every one of these 9 runs — the pattern holds on split_v1 exactly as it did on the ad-hoc split: chunked/causal cache-aware decode costs Nemotron nothing. Beam decode is confirmed **impossible in the streaming path** on any NeMo version tested (`maes` and `malsd_batch` both raise explicit `NotImplementedError` on partial-hypothesis merging, see `results/week_results_2026-07-14_2026-07-17.md` §4) — the offline+beam column above is a ceiling reference, not a deployable number.

Epochs=180 (the numerically best in-domain checkpoint) has **not** yet had this same cross-domain/streaming/beam battery run — only its in-domain merged offline-greedy number is confirmed. Open item.

## Split comparison: same recipe, different legacy result

| | Ad-hoc split (legacy) | split_v1 (legacy, epochs=120) |
|---|---|---|
| Streaming WER | 13.5% | 20.0% |

Legacy diverges sharply by split even holding the recipe fixed — both are N=10 test sets, but this isn't just seed noise (holds across every epoch count tested on split_v1: 90→30.0%, 120→20.0%). Which legacy test set is more representative of real deployment is an open question this repo hasn't resolved.

## Best config found (split_v1, merged, epochs=180, offline greedy 12.11% median / 10.76% best seed)

```
RUN_DATA_MODE=merged
ROOT_DIR=/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1
TRAIN_FILE=<repo-local>/avsr_realtime/labels/splitv1/merged_train_spm1023.csv
VAL_FILE=<repo-local>/avsr_realtime/labels/splitv1/merged_val_spm1023.csv
TEST_FILE=<repo-local>/avsr_realtime/labels/splitv1/merged_test_spm1023.csv
UNFREEZE_ENCODER_LAYERS=-1   # full-FT
TRAIN_DECODER=1              # full-FT, decoder+joint included
BATCH_SIZE=2
GRAD_ACCUM=2                 # effective batch 4
LEARNING_RATE=1e-4           # flat, single LR for the whole model
EPOCHS=180
SPECAUG=1                    # default, never ablated on split_v1
```
All other `nemotron_finetune.py` args left at script defaults: `warmup-steps=200`, `weight-decay=1e-3`, `precision=bf16-mixed`. Neither warmup nor weight decay nor a differential decoder/joint learning rate has been swept on this split — see Open items.

Checkpoints on cluster: `/scratch/pa2753/experiments/nemotron_asr/nemotron_fullft_epochs180_splitv1merged_seed{1,2,3}/nemotron_best.nemo`.

## Open items

- Full cross-domain/streaming/beam battery not yet run on the epochs=180 checkpoints (only epochs=120 has the complete picture above).
- Epoch ceiling not found — 180 was still improving; higher epoch counts untested.
- A differential decoder/joint learning-rate recipe (base LR ~2.5e-5, decoder/joint at ×0.2 of that, weight-decay 0.01, ~70 epochs) was surfaced as a reference config from outside this repo — not implemented here (`nemotron_finetune.py` only supports one flat LR for the whole model currently) and not compared against the flat-LR sweep above. Worth trying: it uses a much lower base LR and a shorter epoch count than anything swept here, so it isn't a strict subset of this sweep's search space.
- Legacy's split-dependent divergence (13.5% ad-hoc vs 20.0% split_v1) unexplained.
- This file's results are not yet reflected in `README.md`'s leaderboard tables (which currently cite epochs=90 ad-hoc-split numbers as the Nemotron entry).
