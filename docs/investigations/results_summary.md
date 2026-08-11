# Auto-AVSR Realtime — Experiment Results Summary

> **Superseded.** This is the earliest results snapshot (2026-06-18), kept for
> historical record only. Two things below are now known wrong: the "41.7%"
> best result was later found to be an unseeded lucky draw from a recipe with
> ~24pp run-to-run variance (honest seeded median is ~50%, see
> `results/week_results_2026-06-23_2026-07-01.md` §10-13), and val_loss-based
> checkpoint selection was later found to anti-correlate with real WER (§4 of
> the same file). For current numbers and methodology, start at the main
> `README.md`'s "Current best results" or `results/week_results_2026-06-23_2026-07-01.md`.

**Date:** 2026-06-18
**Task:** Fine-tune streaming AV-ASR (Emformer RNN-T) on NYU patient lip-reading data
**Metric:** Word Error Rate (WER, lower is better), Real-Time Factor (RTF, must be < 1.0)

---

## Model & Data Setup

| Item | Detail |
|---|---|
| Pretrained model | torchaudio `device_avsr` JIT → eager (`online_avsr_bootstrap.ckpt`) |
| Architecture | 12-layer Emformer RNN-T, segment=32, right-context=4 (~1.44 s latency) |
| Video frontend | LinearVideoFrontend on 44×44 crops |
| Tokenizer | 1023-piece SentencePiece (re-tokenized from units-file IDs) |
| Hardware | NVIDIA H200 / L40S, fp16-mixed AMP |
| Known risk | Pretrained model saw **face crops**; patient data is **mouth-ROI crops** → distribution shift is the primary WER floor |

**Dataset sizes:**

| Split | Train clips | Val clips |
|---|---|---|
| legal_only | 268 | 30 |
| legacy_only | 86 | 10 |
| merged | 354 | 40 |

---

## Results Matrix (Streaming WER)

LoRA fine-tune: r=8, α=16, dropout=0.05, scopes=all, lr=5e-4, batch=2, grad-accum=4.
Best checkpoint per training regime selected by val loss. All evals use `--preprocess roi`, `LEADING_SILENCE_FRAMES=0`.

| Train corpus | Modality | Legal eval (30) | Legacy eval (10) | Merged eval (40) |
|---|---|---|---|---|
| Baseline (no FT) | AV | 98.9% | — | — |
| **Legal** | **AV** | **41.7%** | 95.8% | 55.3% |
| **Merged** | **AV** | 48.1% | 61.7% | 51.5% |
| **Legacy** | **AV** | 95.5% | 87.5% | 93.5% |

**Best result per eval set:**
- Legal eval → Legal train: **41.7%**
- Legacy eval → Merged train: **61.7%**
- Merged eval → Merged train: **51.5%**

---

## Ablation: Face crops vs Mouth-ROI crops (legal_only, AV LoRA)

Controlled comparison — identical config (r=8, α=16, scopes=all, 5700 steps, lr=5e-4)
and identical labels. Only the **video crop type** differs. Face crops were
re-derived from the `_normalized_25p` originals (298/298 cropped, 0 detection failures).

| Run | Crop type | Data root | Legal WER |
|---|---|---|---|
| j11706594 | **Mouth-ROI** | `patient_legal298_crops_unseen` | **42.1%** |
| j11714048 | Face | `legal_only_facecrop` | 59.6% |

**Conclusion: keep mouth-ROI crops.** The handoff hypothesis was that the
face-vs-mouth-ROI distribution shift (the pretrained device_avsr frontend saw
*face* crops) was the primary WER floor, so re-cropping to faces should help.
The experiment shows the **opposite** — mouth-ROI beats face by ~17.5pp.

Likely cause: **effective resolution on the lips.** At 44×44, a face crop spends
most pixels on eyes/forehead/cheeks; the lips (where the lip-reading signal lives)
occupy a small fraction. A mouth-ROI crop fills the frame with lips. Since LoRA
adapts the frontend to whatever it is trained on, "more lip pixels" outweighs
"matches the pretraining distribution." The face-crop alignment idea is dropped.

---

## Full Fine-Tune (legal_only, 268 train clips)

lr=1e-4, batch=2, grad-accum=4, all 24.7 M params trainable.

| Checkpoint | Steps | WER | RTF | Clips w/ output |
|---|---|---|---|---|
| Baseline (no FT) | — | 98.9% | 0.083 | 2/30 |
| `last.ckpt` (epoch 21) | 2850 | 91.8% | 0.071 | 4/30 |
| `epoch=13` (best val_loss=11.81) | ~1750 | 79.8% | 0.069 | 12/30 |
| `model_avg.pth` (avg of 10 epochs) | — | 51.3% | 0.058 | 21/30 |

Full FT val loss was noisy and diverged after epoch 13 (11.81 → 14.45 at epoch 21). LoRA outperforms the best full FT result by ~10pp.

---

## LoRA Fine-Tune — All Runs

| Job | Data | Steps | Legal WER | Notes |
|---|---|---|---|---|
| j11038044 | legal_only | 2850 | **41.7%** | Best overall |
| j11038044 + warmup | legal_only | 2850 | 44.0% | 16 silent frames prepended |
| j11042297 | legal_only | 5700 (resumed) | 47.7% | Overfit — worse than 2850 |
| j11091391 | merged | 2850 | 48.1% | Underfitting — too few steps for larger dataset |
| j11093835 | merged | 5700 | 48.1% | Same as 2850 on legal eval |
| j11097213 | legacy_only | 5700 | 95.5% | Too little data (86 clips) to adapt |

**Key finding:** 2850 steps is the sweet spot for legal_only (268 clips). More steps overfit; merged data at equal steps undertrains.

---

## Streaming vs Utterance Mode

Utterance mode = full-sequence beam search (offline upper bound). Streaming = chunked Emformer with carry-state.

| Model | Eval set | Streaming WER | Utterance WER |
|---|---|---|---|
| Legal LoRA (j11038044) | Legal | 41.7% | 41.8% |
| Merged LoRA (j11093835) | Merged | 51.5% | 56.3% |

**Streaming is not losing anything vs utterance mode.** The Emformer carry-state approach is well-calibrated — there is no offline upper-bound gap to recover by switching to non-streaming decoding.

---

## Key Findings

**1. LoRA >> Full Fine-Tune**
Best LoRA (41.7%) beats best full FT (51.3% with model_avg) by ~10pp. With only 268 training clips, full FT collapses to silent output on most clips. LoRA's constrained parameter space prevents this.

**2. Legal train generalizes poorly to legacy (and vice versa)**
Legal→Legacy: 95.8%. Legacy→Legal: 95.5%. The two datasets have very different recording conditions. Merged training is the only thing that bridges them (61.7% on legacy vs 95.8% for legal-only).

**3. Legacy-only training is ineffective**
86 training clips is too few — the legacy LoRA achieves 87.5–95.5% WER across all eval sets, barely better than baseline.

**4. Warmup effect is inconsistent**
16-frame silent warmup improves some runs but hurts the best run (41.7% → 44.0%). The sentence-initial word-drop pattern is a structural streaming model limitation, not a cold-state artifact.

**5. RTF is never a concern**
All RTFs: 0.058–0.083, well under the 1.0 real-time threshold.

---

## Dominant Error Pattern — Best Run (Legal LoRA, 41.7% WER)

The model reliably decodes the sentence body but drops sentence-initial words ("but", "and", "can") which fall in the first Emformer segment with no prior context.

| Reference | Hypothesis | WER |
|---|---|---|
| **but i** do not know the schedule for the trial | i do not know the schedule for the trial | 0.10 |
| **but the** case is not on trial now | the case is not on trial now | 0.12 |
| **but i** did not think about that | i did not think about that | 0.14 |
| **and they** will confirm the trial | they will confirm the trial | 0.17 |
| **but i** do not know that | i do not know that | 0.17 |
| **but i** will not do that | i will not do that | 0.17 |
| **but my** client is not there | my client is not there | 0.17 |

Only 2/30 clips produced empty output (vs 18/30 for full FT, 28/30 for baseline).

---

## Comparison with Offline System (Conformer+CTC)

The offline system (Conformer+CTC, closed-vocab constrained decoding) achieved significantly lower WER. The streaming model is roughly 2× worse across equivalent conditions.

| Train | Eval | Offline (AV, constrained) | Streaming LoRA (AV, open) |
|---|---|---|---|
| Legal | Legal | 23.8% | 41.7% |
| Merged | Legal | 19.5% | 48.1% |
| Merged | Legacy | 27.5% | 61.7% |
| Merged | Merged | 21.3% | 51.5% |

The gap is expected: (1) face-crop pretraining vs mouth-ROI evaluation, (2) smaller model capacity, (3) streaming latency constraint vs offline full-sequence decoding.

---

## Checkpoints

| Description | Path |
|---|---|
| Pretrained bootstrap | `/home/pa2753/auto-avsr-realtime/cpts/online_avsr_bootstrap.ckpt` |
| Full FT best epoch | `.../avsr_realtime_finetune/finetune_legal_only_av_j10962142_*/epoch=13-val_loss=11.8071.ckpt` |
| Full FT model avg | `.../avsr_realtime_finetune/finetune_legal_only_av_j10962142_*/model_avg.pth` |
| **LoRA best — Legal train (41.7%)** | `.../avsr_realtime_lora/lora_all_legal_only_av_j11038044_1781747649/model_lora_merged.pth` |
| LoRA — Merged train | `.../avsr_realtime_lora/lora_all_merged_av_j11093835_1781808685/model_lora_merged.pth` |
| LoRA — Legacy train | `.../avsr_realtime_lora/lora_all_legacy_only_av_j11097213_1781814366/model_lora_merged.pth` |

All scratch paths under `/scratch/pa2753/experiments/`.
