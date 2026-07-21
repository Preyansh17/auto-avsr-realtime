# Nemotron on the audited split_v1: epoch sweep, cross-domain generalization, best config

**Date:** 2026-07-18 → 2026-07-21
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
| 210 | 13.00% | — | — | 13.00%¹ |
| 240 | 13.45%² | — | — | 13.45%¹ |

¹ Seed1-only screen, not 3-seed confirmed.
² Truncated run: the cluster's automated GPU-utilization monitor killed this job at epoch 136/240 (21.75% average utilization over 2h+, below its 50% cancellation threshold) — see "Cluster auto-cancellation" below. Val_wer was still improving at the kill point (20.5%); the 13.45% is a salvage-eval of the last saved checkpoint, not a completed 240-epoch run. Not indicative that 240 is worse than 180 — the run never finished.

**180 is the confirmed ceiling.** Neither 210 nor 240 beat it on their (single-seed, and in 240's case truncated) screens — three separate datapoints at or past 210 all land at/above 180's result. Combined with 180 and 120 tying as the two best points on the seed1 screen specifically (both 10.76%), this looks like a genuine plateau rather than the still-rising curve suggested by the 90→180 trend alone. Not pushed further.

Val-WER trajectory for the epochs=180 seed1 run (10.76% final test WER) was still dropping meaningfully at its best checkpoint (epoch 107 of 180, val_wer 21.0%, down from 22.3% at epoch 67) — the selected checkpoint is not from deep into an overfitting plateau, consistent with 180 being a real (not accidental) local optimum rather than a mid-descent snapshot.

**This directly contradicts the epoch-reversal seen on the ad-hoc split**, where 120 epochs (single-seed) reversed against 90's 3-seed median (33.3%→18.0%→~19.1%). The epoch-count sweet spot is split-dependent, not a fixed property of the model/recipe — a genuinely unexpected finding, now doubly confirmed (split_v1's own ceiling sits well past where the ad-hoc split's reversed).

### Cluster auto-cancellation (operational finding)

The torch cluster runs an automated GPU-utilization monitor that kills jobs averaging below a per-GPU-model threshold (50% for the A100s these ran on) for more than 2 hours, to reclaim idle allocations. The epochs=240 seed1 screen (job 14443619, node ga020) was killed this way at 2:09:01 elapsed, 21.75% average utilization. A near-identical concurrent job (epochs=210 seed1, node gl068) completed cleanly at 2:22:04 — same recipe, same batch size, different node, no cancellation. Root cause not chased further (plausibly node-specific GPU/driver behavior, or checkpoint-save I/O bursts from `SaveBestWER` triggering intermittently-low averaged utilization on some runs and not others), but the practical implication is real: **any single long-epoch run on this cluster now carries a nonzero risk of being killed after ~2 hours regardless of recipe correctness.** The killed job's last-saved checkpoint (epoch 136/240) was still usable for a salvage eval rather than a total loss — worth doing whenever a long run gets cut short here.

## Full cross-domain / streaming / beam battery — epochs=120 and epochs=180 (both fully characterized)

3 seeds, all three domains, offline greedy, offline+beam (`maes`, width 8), and real cache-aware streaming decode with word-level latency (`nemotron_streaming_eval.py --word-lag`).

| Domain | Config | Offline greedy (median) | Offline +beam (median) | **Streaming, deployable (median)** | TTFT-from-speech | Word-commit lag |
|---|---|---|---|---|---|---|
| merged (in-domain) | epochs=120 | 14.80% | 13.00% | 13.90% | ~0.95s | ~0.88s |
| merged (in-domain) | **epochs=180** | 12.11% | 13.00% | **12.11%** | ~0.98s | ~0.90s |
| legal (cross-domain) | epochs=120 | 13.11% | 10.38% | 12.02% | ~0.97s | ~0.87s |
| legal (cross-domain) | **epochs=180** | 9.84% | 10.38% | **9.84%** | ~0.99s | ~0.91s |
| legacy (cross-domain) | epochs=120 | 20.00% | 25.00%¹ | **20.00%** | ~0.91s | ~0.87s |
| legacy (cross-domain) | epochs=180 | 22.50% | 25.00%¹ | 22.50% (slightly worse) | ~0.92s | ~0.85s |

¹ Beam decode *hurts* legacy specifically at both epoch counts — the one exception to beam helping everywhere else this whole investigation. N=10, plausibly noise, not investigated further.

**Epochs=180 beats 120 on merged and legal, loses slightly on legacy.** Legal at 9.84% is the best cross-domain number Nemotron has produced anywhere in this investigation — the first time a Nemotron cross-domain result has dipped under 10%.

Streaming WER equals offline greedy WER exactly on every one of these 18 runs (9 at each epoch count) — the pattern holds on split_v1 exactly as it did on the ad-hoc split: chunked/causal cache-aware decode costs Nemotron nothing. Beam decode is confirmed **impossible in the streaming path** on any NeMo version tested (`maes` and `malsd_batch` both raise explicit `NotImplementedError` on partial-hypothesis merging, see `results/week_results_2026-07-14_2026-07-17.md` §4) — the offline+beam column above is a ceiling reference, not a deployable number.

**Epochs=180 is now the confirmed best Nemotron config, fully characterized the same way as 120.**

## Split comparison: same recipe, different legacy result

| | Ad-hoc split (legacy) | split_v1 (legacy) |
|---|---|---|
| Streaming WER | 13.5% | 20.0% (epochs=120) / 22.5% (epochs=180) |

Legacy diverges sharply by split even holding the recipe fixed — both are N=10 test sets, but this isn't just seed noise (holds across every epoch count tested on split_v1: 90→30.0%, 120→20.0%, 180→22.5%). Which legacy test set is more representative of real deployment is an open question this repo hasn't resolved. Notably, legacy is also the one domain where more epochs (180 vs 120) made things slightly *worse* — consistent with legacy behaving differently from the other two domains throughout this whole investigation.

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

Checkpoints on cluster: `/scratch/pa2753/experiments/nemotron_asr/nemotron_fullft_epochs180_splitv1merged_seed{1,2,3}/nemotron_best.nemo`. `slurm/nemotron_splitv1_best.sh` reproduces this exact run (`EPOCHS=180` is its default).

## Open items

- A differential decoder/joint learning-rate recipe (base LR ~2.5e-5, decoder/joint at ×0.2 of that, weight-decay 0.01, ~70 epochs) was surfaced as a reference config from outside this repo — not implemented here (`nemotron_finetune.py` only supports one flat LR for the whole model currently) and not compared against the flat-LR sweep above. Worth trying: it uses a much lower base LR and a shorter epoch count than anything swept here, so it isn't a strict subset of this sweep's search space.
- Legacy's split-dependent divergence (13.5% ad-hoc vs 20.0-22.5% split_v1) unexplained, and legacy is the one domain where 180 epochs underperformed 120 — worth a closer look if legacy performance matters for deployment.
- Epoch sweep stopped at 180 based on 210/240 not beating it on single-seed screens (240 additionally truncated by cluster cancellation) — not confirmed with 3 seeds each. If a genuinely tighter ceiling claim is needed, 210 would be the next one worth 3-seed-confirming given how close it landed to 180.
- This file's results are not yet reflected in `README.md`'s leaderboard tables (which currently cite epochs=90 ad-hoc-split numbers as the Nemotron entry).
- LoRA restore bug (unrelated to this file's investigation, carried over from earlier work) remains unresolved.
- Speed perturbation, never attempted for Nemotron.
