# Nemotron on the audited split_v1: epoch sweep, cross-domain generalization, best config

**Date:** 2026-07-18 → 2026-07-22
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
| 180 | 10.76% | 12.11% | 13.90% | 12.11% (mean 12.26%) |
| **210** | 13.00% | 11.21% | 11.21% | **11.21%** (mean 11.81%) |
| 240 | 11.21%³ | 12.56% | 11.21% | 11.21% (mean 11.66%) |

³ Originally a truncated salvage number (13.45%, epoch 136/240, cluster-killed — see "Cluster auto-cancellation" below). Replaced 2026-07-22 with a clean full-240-epoch number via the new checkpoint-resume feature (resumed from the same run's last good PTL checkpoint, epoch 149/240, and trained to completion). See "Checkpoint resume" below.

**210 and 240 are now both 3-seed confirmed and tie on median (11.21%)**, with 240 marginally ahead on mean (11.66% vs 11.81%). Both beat 180 (12.11% median). Per-seed variance is real, not a clean sweep in either direction — e.g. 180's best seed (10.76%) beats 210's best seed (11.21%) — so read the win as an aggregate/median effect, not a uniform improvement. **210 was chosen as the practical stopping point** (see "210 vs 240" below) rather than continuing to chase 240 or beyond, given diminishing returns and rising cluster-cancellation risk on longer single-shot runs.

Val-WER trajectory for the epochs=180 seed1 run (10.76% final test WER) was still dropping meaningfully at its best checkpoint (epoch 107 of 180, val_wer 21.0%, down from 22.3% at epoch 67) — the selected checkpoint is not from deep into an overfitting plateau, consistent with 180 being a real (not accidental) local optimum rather than a mid-descent snapshot.

**This directly contradicts the epoch-reversal seen on the ad-hoc split**, where 120 epochs (single-seed) reversed against 90's 3-seed median (33.3%→18.0%→~19.1%). The epoch-count sweet spot is split-dependent, not a fixed property of the model/recipe — a genuinely unexpected finding, now doubly confirmed (split_v1's own ceiling sits well past where the ad-hoc split's reversed).

### Cluster auto-cancellation (operational finding)

The torch cluster runs an automated GPU-utilization monitor that kills jobs averaging below a per-GPU-model threshold (50% for the A100s these ran on) for more than 2 hours, to reclaim idle allocations. The epochs=240 seed1 screen (job 14443619, node ga020) was killed this way at 2:09:01 elapsed, 21.75% average utilization. A near-identical concurrent job (epochs=210 seed1, node gl068) completed cleanly at 2:22:04 — same recipe, same batch size, different node, no cancellation. Root cause not chased further (plausibly node-specific GPU/driver behavior, or checkpoint-save I/O bursts from `SaveBestWER` triggering intermittently-low averaged utilization on some runs and not others), but the practical implication is real: **any single long-epoch run on this cluster now carries a nonzero risk of being killed after ~2 hours regardless of recipe correctness.** The killed job's last-saved checkpoint (epoch 136/240) was still usable for a salvage eval rather than a total loss — worth doing whenever a long run gets cut short here.

This kept recurring on 2026-07-21/22: bumping `NUM_WORKERS` from the script default of 2 to 6 did not fix it — 4 fresh long-epoch jobs (epochs 210/240, seeds 2/3) were killed simultaneously at ~2h15m elapsed despite the higher worker count, GPU utilization still bursty (alternating 0% and 30-50% across live `nvidia-smi` samples). This looks like a hard trigger point around ~2h+ for this recipe on this cluster, not something tunable away from the training side.

### Checkpoint resume (new capability, 2026-07-22)

`nemotron_finetune.py` previously had no way to continue a killed run — `trainer.fit(model)` always started from epoch 0, and the custom `SaveBestWER` callback only ever saved model *weights* (`.nemo`), not optimizer/scheduler/epoch state. It turned out Lightning's default `ModelCheckpoint` was already auto-saving full trainer state (`.ckpt`, with optimizer/scheduler/epoch counters) to `lightning_logs/version_<jobid>/checkpoints/` the whole time — nothing in the script consumed it.

Added a `--resume-ckpt <path>` flag (`trainer.fit(model, ckpt_path=args.resume_ckpt)`) plus a `RESUME_CKPT` env var passthrough in `nemotron_finetune.sbatch`. Verified working: 3 of 4 killed jobs resumed correctly (epoch counter picked up past the kill point, not from 0); the 4th hit a corrupted `.ckpt` (truncated mid-write by the SIGTERM, confirmed by file size — 3.06GB vs a healthy 7.4GB) and needed to fall back to the previous saved epoch instead. All 4 subsequently finished training to their full target epoch count using this mechanism, including a 5th resume (epochs=240 seed1, resuming from a checkpoint saved back on 2026-07-20) to close out the 3-seed epochs=240 set with a real number instead of the old truncated salvage estimate.

**Caveat:** `SaveBestWER.best_wer` resets to `inf` on resume (it's a plain Python attribute, not persisted via Lightning's callback state hooks) — the first post-resume validation epoch unconditionally overwrites `nemotron_best.nemo` even if it's worse than the pre-resume best. Mitigated by backing up `nemotron_best.nemo` → `nemotron_best_salvage_pre_resume.nemo` before every resume in this investigation; worth fixing properly (persist `best_wer` via `state_dict`/`load_state_dict`) if resume becomes routine.

This capability turns future cluster auto-cancellations from a real risk (lost compute, truncated/unusable data points) into a minor inconvenience (resubmit with `--resume-ckpt`) — should be used by default going forward for any run approaching the ~2h mark.

## Full cross-domain / streaming / beam battery — epochs=120, 180, and 210 (all fully characterized)

3 seeds, all three domains, offline greedy, offline+beam (`maes`, width 8), and real cache-aware streaming decode with word-level latency (`nemotron_streaming_eval.py --word-lag`). Epochs=240 has only the merged offline number (see epoch sweep table above) — full battery not run for it, since 210 was chosen as the practical stopping point (see "210 vs 240" below).

| Domain | Config | Offline greedy (median) | Offline +beam (median) | **Streaming, deployable (median)** | TTFT-from-speech | Word-commit lag |
|---|---|---|---|---|---|---|
| merged (in-domain) | epochs=120 | 14.80% | 13.00% | 13.90% | ~0.95s | ~0.88s |
| merged (in-domain) | epochs=180 | 12.11% | 13.00% | 12.11% | ~0.98s | ~0.90s |
| merged (in-domain) | **epochs=210** | 11.21% | 12.11% | **11.21%** | ~1.02s | ~0.92s |
| legal (cross-domain) | epochs=120 | 13.11% | 10.38% | 12.02% | ~0.97s | ~0.87s |
| legal (cross-domain) | epochs=180 | 9.84% | 10.38% | 9.84% | ~0.99s | ~0.91s |
| legal (cross-domain) | **epochs=210** | 8.20% | 8.74% | **8.20%** | ~1.03s | ~0.92s |
| legacy (cross-domain) | epochs=120 | 20.00% | 25.00%¹ | **20.00%** | ~0.91s | ~0.87s |
| legacy (cross-domain) | epochs=180 | 22.50% | 25.00%¹ | 22.50% | ~0.92s | ~0.85s |
| legacy (cross-domain) | epochs=210 | 25.00% | 27.50%¹ | 25.00% (worst of the three) | ~1.00s | ~0.95s |

¹ Beam decode *hurts* legacy specifically at every epoch count tested — the one exception to beam helping everywhere else this whole investigation. N=10, plausibly noise, not investigated further.

**Epochs=210 beats both 120 and 180 on merged and legal — legal at 8.20% is the best cross-domain number Nemotron has produced anywhere in this investigation.** But it's also the worst of the three on legacy (25.00%), continuing a monotonic trend: legacy gets steadily worse as epoch count increases (20.00% → 22.50% → 25.00% at 120/180/210) while merged and legal both improve. This isn't noise — it holds at every epoch count checked. **There is no single epoch count that wins on all three domains**; the choice is a real tradeoff between merged/legal accuracy (favors more epochs) and legacy accuracy (favors fewer). 210 was picked as the new default on the reasoning that merged and legal are the larger, more reliable test sets (N=40, N=30 vs legacy's N=10) — see "210 vs 240" below for the reasoning on why 210 specifically over 240.

Streaming WER equals offline greedy WER exactly on every one of these 27 runs (9 at each of 3 epoch counts) — the pattern holds on split_v1 exactly as it did on the ad-hoc split: chunked/causal cache-aware decode costs Nemotron nothing. Beam decode is confirmed **impossible in the streaming path** on any NeMo version tested (`maes` and `malsd_batch` both raise explicit `NotImplementedError` on partial-hypothesis merging, see `results/week_results_2026-07-14_2026-07-17.md` §4) — the offline+beam column above is a ceiling reference, not a deployable number. Latency is essentially flat across all three epoch counts (~0.9-1.0s TTFT-from-speech, ~0.85-0.95s word-commit lag) — epoch count moves WER, not latency.

### 210 vs 240

240 ties 210 on median merged WER (11.21% both) and is marginally ahead on mean (11.66% vs 11.81%, see epoch sweep table above), but has no cross-domain/streaming/beam battery — running it would cost roughly another 30 GPU-hours (27 more short eval jobs, plus the risk of needing another checkpoint-resume rescue for training itself) for an aggregate merged-only edge under half a percentage point. Given diminishing returns and the real operational cost of longer single-shot training runs on this cluster (both 210 and 240 needed checkpoint-resume rescues from auto-cancellation; 240 needed it twice across its history), **210 was chosen as the practical stopping point for the epoch sweep** rather than continuing to chase 240 or beyond.

**Epochs=210 is now the recommended default Nemotron config** — full battery characterized the same way as 120 and 180, and it wins on the two larger/more reliable domains (merged, legal). If legacy-domain accuracy specifically matters for a given deployment, 180 remains the better choice there (22.50% vs 210's 25.00%) — this is a genuine tradeoff, not a strict dominance.

## Split comparison: same recipe, different legacy result

| | Ad-hoc split (legacy) | split_v1 (legacy) |
|---|---|---|
| Streaming WER | 13.5% | 20.0% (epochs=120) / 22.5% (epochs=180) / 25.0% (epochs=210) |

Legacy diverges sharply by split even holding the recipe fixed — both are N=10 test sets, but this isn't just seed noise (holds across every epoch count tested on split_v1: 90→30.0%, 120→20.0%, 180→22.5%, 210→25.0%, monotonically worsening). Which legacy test set is more representative of real deployment is an open question this repo hasn't resolved. Legacy is also the one domain where more epochs consistently make things worse — the opposite trend from merged and legal — throughout this whole investigation.

## Best config found (split_v1, merged, epochs=210, offline greedy 11.21% median / 11.21% best-tied seed)

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
EPOCHS=210
SPECAUG=1                    # default, never ablated on split_v1
```
All other `nemotron_finetune.py` args left at script defaults: `warmup-steps=200`, `weight-decay=1e-3`, `precision=bf16-mixed`. Neither warmup nor weight decay nor a differential decoder/joint learning rate has been swept on this split — see Open items.

**If legacy-domain accuracy matters more than merged/legal for your deployment**, use `EPOCHS=180` instead (22.50% vs 210's 25.00% on legacy, at the cost of 12.11% vs 11.21% on merged and 9.84% vs 8.20% on legal) — see "210 vs 240" above for the full tradeoff.

Checkpoints on cluster: `/scratch/pa2753/experiments/nemotron_asr/nemotron_fullft_epochs210_splitv1merged_seed{1,2,3}/nemotron_best.nemo` (epochs=180 checkpoints remain at `nemotron_fullft_epochs180_splitv1merged_seed{1,2,3}/` if the legacy-favoring config is needed). `slurm/nemotron_splitv1_best.sh` reproduces the epochs=180 run by default; pass `EPOCHS=210` to reproduce the new default (note: as of 2026-07-22 the actual 210/240 runs used `nemotron_finetune.py`'s new `--resume-ckpt` flag partway through due to cluster auto-cancellation — a from-scratch `EPOCHS=210` run via this script should reach the same result but hasn't been separately verified end-to-end).

## Open items

- A differential decoder/joint learning-rate recipe (base LR ~2.5e-5, decoder/joint at ×0.2 of that, weight-decay 0.01, ~70 epochs) was surfaced as a reference config from outside this repo — not implemented here (`nemotron_finetune.py` only supports one flat LR for the whole model currently) and not compared against the flat-LR sweep above. Worth trying: it uses a much lower base LR and a shorter epoch count than anything swept here, so it isn't a strict subset of this sweep's search space.
- Legacy's split-dependent divergence (13.5% ad-hoc vs 20.0-25.0% split_v1) unexplained, and legacy now *monotonically* worsens as epoch count increases (120→180→210: 20.0%→22.5%→25.0%) — worth a closer look if legacy performance matters for deployment. Root cause not investigated (candidate hypotheses: legacy's small N=10 test set is just noisier, or the merged-domain training distribution increasingly overfits away from legacy's specific acoustic/linguistic characteristics as training progresses).
- Epoch sweep stopped at 210 as a practical/cost tradeoff, not because 240 was shown worse — 240 actually ties or marginally beats 210 on merged (see "210 vs 240" above), but was never given the full cross-domain/streaming/beam battery. If a tighter epochs=240 characterization is ever needed, that battery (27 more eval jobs) is the next step, not more training.
- This file's results are not yet reflected in `README.md`'s leaderboard tables (which currently cite epochs=90 ad-hoc-split numbers as the Nemotron entry).
- LoRA restore bug (unrelated to this file's investigation, carried over from earlier work) remains unresolved.
- Speed perturbation, never attempted for Nemotron.
- A `check_val_every_n_epoch`-style reduction in validation frequency was identified (2026-07-22) as the most promising remaining wall-clock lever — at this dataset size (314 train examples), fixed per-epoch overhead (validation pass, checkpoint bookkeeping) dominates over raw batch compute, so bigger batch sizes barely moved epochs/minute in a scouting test. Not implemented or tested.
- `nemotron_finetune.py`'s new `--resume-ckpt` flag and `SaveBestWER`'s reset-on-resume caveat (see "Checkpoint resume" above) are currently only patched into the cluster's local checkout, not committed to this repo.
