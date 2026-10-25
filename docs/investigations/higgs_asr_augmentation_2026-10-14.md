# Higgs synthetic speech as ASR augmentation for the patient, 2026-10-14

**Question.** Does synthetic patient speech from the finetuned Higgs Audio v2 adapter (B_r64 epoch 8,
`higgs_lora_finetune_2026-10-06.md`) lower ASR word error rate on the real split_v1 merged test set (40 clips)?

**Answer.** No detectable effect. In the pre-registered confirmatory test on 7 fresh seeds, no Whisper large-v3 arm
(real + synthetic at 0.5x, pool A only, duration-filter only) beat real-only. That holds for the val-selected checkpoint
(primary) and for the final-epoch checkpoint (val-free secondary). The 3-seed leads that motivated the test (A0.5 -1.79,
D0.5 -1.94 pp) did not replicate. On Nemotron, the M0.5 gain shrinks to -1.64 pp with an interval crossing 0 once
checkpoint selection is equalised. Synthetic-only training is worse than real-only for Whisper; for Nemotron the +5 pp
rests on a 3-seed Welch test and fails the sentence-level tests. Power is limited: R's seed-to-seed sd is ~1.5 pp over
40 sentences, and the pre-registered test has an 80% minimum detectable effect of ~4-5 pp (corrected 2026-10-20; the
original "~2 pp" was too optimistic). **Caveat added 2026-10-21:** every Whisper number here was decoded with a prompt
the model was not trained on, and that mismatch shifts the arm deltas in R's favour (see "Design and inference review").

Code, design, pre-registration and every table: `asr_aug/README.md`; cluster mirror `/scratch/pa2753/asr_aug/`
(reports in `data/report_*.md`).

## Setup

- Text: pool A = 1000 new sentences from the 76-word ASR-train vocabulary; pool B = 1000 open-vocabulary everyday
  sentences (304 word types). Text leakage guard: 0 sentences equal or within 1 word edit of any val/test sentence; 0
  contain a val/test-only word (bad, but, hello, music, tired).
- Audio: 2000 clips, one per sentence, references rotated over ft_stitch5 / stitch5 / lowpitch5 / median1.
- Filter: duration and silence rules, plus an ASR judge (real-only Whisper seed 1). Kept A 894, B 647.
- Whisper: large-v3 best recipe (full FT + SpecAugment, lr 1e-5, 30 epochs, val-WER checkpoint selection).
  Arms: R, S (synthetic only), M0.25-M4 (real + r x 314 synthetic, A/B half each), and ablations A0.5, B0.5, D0.5.
- Nemotron: split_v1 best recipe, 210 epochs (S: 43 epochs = R's step count). Arms R, M0.5, S, 3 seeds each.

## Results

Whisper test WER (%), mean over seeds; delta vs R with sentence-bootstrap 95% interval.

| | R | M0.5 | A0.5 | D0.5 |
|---|---|---|---|---|
| seeds 1-3 (exploratory) | 7.62 | 7.03 (-0.60) | 5.83 (-1.79) | 5.68 (-1.94 [-3.89, -0.30]) |
| **seeds 4-10, best/ (primary)** | 7.43 | 8.78 (+1.35) | 6.92 (-0.51) | 7.56 (+0.13) |
| seeds 4-10, final epoch | 8.07 | 6.47 (-1.60) | 6.34 (-1.73) | 6.79 (-1.28) |

Primary decision (two-level bootstrap + Holm over 3 arms): no arm helps (p = 0.34 / 0.76 / 0.95). Post-hoc seed-level
Welch tests agree; the smallest is A0.5 final epoch at p = 0.040, which does not survive Holm. Other 3-seed arms:
S 12.41 (+4.78 [+1.63, +8.10]), M0.25 9.27, M1 7.47, M2 7.77, M4 7.32, B0.5 8.37.

Nemotron test WER (%), seeds 1-3:

| checkpoint | R | M0.5 | S |
|---|---|---|---|
| best.nemo (R used the pre-fix selection rule) | 10.61 | 8.67 (-1.94) | 15.70 (+5.08) |
| final epoch (no selection) | 11.21 | 9.57 (-1.64 [-3.79, +0.57]) | 16.29 (+5.08) |

## What the audit found

An adversarial audit ran six lenses, each finding re-checked by two independent skeptics.

- **Test is clean.** No test clip, sentence or test-only word reached the TTS training data, the reference prompts, the
  sentence pools or any ASR training set. Pool A is not closer to test than real train.
- **Val is not independent of the synthetic data (major).** `tts_finetune/prep_data.py` drops only ASR-test sentences, so
  37 of 40 ASR val clips are in the adapter's tts_train. The ft_stitch5 reference also splices in one val clip. Val picks
  each checkpoint, r* and the judge thresholds. This is why the final-epoch secondary analysis was added before any seed
  4-10 result existed. R does better at its val-selected checkpoint than at its final epoch; the augmented arms do better
  at their final epoch. That pattern fits val misleading selection for augmented runs. Future TTS-for-ASR work must
  exclude ASR val from TTS training.
- **Pool-B judge selects on vocabulary (major).** A 76-word judge cannot transcribe pool-B words, so the CER rule keeps
  train-like pool-B text, and comparisons of the judge filter against the duration-only filter mix vocabulary with
  audio quality.
- **Nemotron R baseline used the pre-fix SaveBestWER (critical, for Nemotron only).** Resolved by the selection-free
  decode above.
- **SaveBestWER one-epoch lag (major, all Nemotron runs).** Fixed in `asr_baselines/nemotron_finetune.py`
  (`on_validation_end`). Effective Nemotron LR is a flat 1e-6, not 1e-4.
- **Statistics (verified 2026-10-16).** The two-level bootstrap double-counts the seed x sentence interaction: its SE
  is 1.2-1.6x too wide, and corrected p values stay >= 0.17. The conclusion does not change.

## Design and inference review (2026-10-20)

The lens that the usage limit cut off ran on 2026-10-20; each finding was re-checked by a verifier. Confirmed findings,
most severe first:

- **Whisper decoder-prompt mismatch (major, all Whisper numbers in both studies).** `whisper_finetune.py` never called
  `set_prefix_tokens`, and its collator stripped `<|endoftext|>` instead of `<sot>`. So training saw
  `<sot><sot><notimestamps>`, while every decode (val selection and test) used `<sot><en><transcribe><notimestamps>`.
  Re-decoding all 49 surviving best/ checkpoints with the trained prompt (`asr_aug/prefix_redecode.sbatch`,
  `prefix_report.py`) costs every arm WER but costs R most, so the published prompt favours R:

  | test WER, best/ | R | M0.5 | A0.5 | D0.5 | TORGO R | TORGO M0.5 | TORGO M0.5sc |
  |---|---|---|---|---|---|---|---|
  | published prompt | 7.43 | 8.78 | 6.92 | 7.56 | 33.47 | 33.09 | 32.10 |
  | trained prompt | 10.25 | 9.42 | 8.78 | 8.90 | 37.97 | 35.76 | 33.61 |
  | delta vs R, published | | +1.35 | -0.51 | +0.13 | | -0.38 | -1.37 |
  | delta vs R, trained | | -0.83 | -1.47 (p 0.053) | -1.35 (p 0.096) | | -2.21 (p 0.15) | -4.35 (p 0.013, [-7.21, -2.31]) |

  Under the trained prompt the patient deltas flip sign, but none passes both pre-registered tests. TORGO M0.5sc
  would, but it is a post-hoc arm decoded post hoc. The prompt choice itself is post hoc, so the pre-registered
  decision stands as stated. The finding is that the null is not robust to a decode artefact that favours R. A
  checkpoint trained with the matched prompt (`--label-prefix lang_task`, the new default) is worse: R_lt seeds 4-6
  give 11.06 +- 3.24 vs 7.43. The mismatch was regularising. The re-voicing study (`asr_aug/revoice/README.md`)
  runs both recipes.
- **Mislabelled 180 s TORGO test clip (major, TORGO Nemotron).** M01/Session2_3/0180.wav is labelled "hill" but holds
  many prompts. It adds 45-68 Nemotron errors per run (~3.5 pp of test WER). Without it: Nemotron R 25.92, final-epoch
  M0.5 - R = -1.81 (Welch p 0.039, [-2.97, -0.72]), best.nemo -0.31 (p 0.65). The decision stands, because best.nemo is
  still null. Nemotron beats Whisper by ~7 pp on TORGO, not ~4. `build_splits.py` has no upper duration bound.
- **Power overstated (major).** See the corrected sentence in the answer above. On TORGO the Welch rule has ~34% power
  for 2 pp, so "effects of roughly 2 pp would have been visible" is wrong; read it as ~3.5-4 pp.
- **Minor:**
  - One greedy repetition loop (218 words, TORGO M0.5 seed 6, final epoch) moves the TORGO Whisper final-epoch delta
    from +1.70 to -0.27; the decode has no loop guard.
  - TORGO Whisper best/ is epoch 1 in 18 of 21 runs, so the val optimum is not bracketed.
  - The normaliser does not expand contractions: patient WER is 0.2-0.45 pp high, and deltas move <= 0.13 pp.
  - Ten TORGO test words occur inside train sentences (69/396 word clips), so "every test single word is unseen" is
    false in that sense.
  - TORGO synthetic text excluded test vocabulary, so the TORGO null is about acoustic augmentation without the target
    words.
  - Seeds 4-10 show no val advantage for augmented arms, so "val misleads selection" is not shown.
  - The confirmatory arm used one fixed 157-clip synthetic subset, not per-epoch resampling.
  - M0.25 was added after test results and is overclaimed in the README.
  - Nemotron SEED does not control data order (NeMo reseeds to 0 and shuffles with a true-random seed).
  - Nemotron batch padding does not change decodes (batch 1 vs 8: 0/80 hyps differ, checked 2026-10-21).

## Infrastructure note

The first seed 4-10 launch (31 concurrent Whisper large-v3 runs, each holding 52-69 GB of checkpoints) filled the 5 TB
/scratch quota and every job failed. `asr_aug/whisper_launch.sh` now runs them in at most 8 lanes.

## TORGO extension (step 6, 2026-10-16)

Design (`asr_aug/torgo/README.md`, pre-registered 2026-10-15): 8 TORGO dysarthric speakers pooled; split by prompt text
(train 1693 / val 590 / test 539 clips, no shared text, so every test single word is unseen); per-speaker Higgs adapters
trained on ASR-train clips only (fixes the patient val contamination); 160 words + 60 sentences per speaker; zero-shot
large-v3 judge calibrated on real train; M0.5 = real + 0.5x synthetic per speaker. Whisper 7 seeds, Nemotron 3 seeds.

| model / checkpoint | R | M0.5 | delta [95%] | Welch p |
|---|---|---|---|---|
| Whisper large-v3, best/ (primary) | 33.47 | 33.09 | -0.38 [-1.48, +0.73] | 0.75 |
| Whisper, final epoch | 43.99 | 45.69 | +1.70 [-1.43, +6.56] | 0.50 |
| Nemotron, best.nemo | 29.44 | 29.71 | +0.27 [-1.21, +2.08] | 0.71 |
| Nemotron, final epoch | 29.66 | 28.52 | -1.14 [-2.72, +0.80] | 0.18 |

No detectable effect on either model, with a test set 13x larger than the patient's (Whisper seed sd 1.6-2.6 pp; the
detectable effect is ~3.5-4 pp, not 2). Zero-shot large-v3 scores 52.55 (it hallucinates on isolated words). Post hoc
(2026-10-19): with the silence filter recalibrated, M0.5sc best/ is -1.37 [-2.37, -0.36] but Welch p 0.24, so it also
fails the rule. Nemotron numbers include the mislabelled 180 s clip (see the review).

**Conclusion across both studies (revised 2026-10-21):** at 0.5x, synthetic speech from per-speaker finetuned Higgs v2
gave no effect detectable at ~4 pp on the pre-registered tests, for Whisper or Nemotron. Synthetic-only training hurts
Whisper on the patient set. The Whisper null is not robust: the decode-prompt artefact favours R, and with the trained
prompt the augmented arms lead by 0.8-4.4 pp.

## Re-voicing (design B, 2026-10-20 to 2026-10-23)

Each of the 303 eligible ASR-train clips (11 that say a val sentence skipped) was re-voiced twice by a clean adapter
(trained with every ASR val and test sentence removed), using the clip itself as the reference and its own transcript
as the text. The filter kept 550 of 606. V1 = 314 real + 314 re-voiced; Vall = 314 + 550. Pre-registration, amendments
and job IDs: `asr_aug/revoice/README.md`. Test WER % on the 40 test sentences:

| comparison | checkpoint | arm | ref | delta [95%] | Welch p |
|---|---|---|---|---|---|
| **Primary:** Whisper V1_lt vs R_lt (fixed prompt), seeds 4-10 | best/ | 9.29 | 10.70 | -1.41 [-5.19, +1.13] | 0.21 |
| | final epoch | 11.15 | 10.70 | +0.45 [-2.42, +2.86] | 0.66 |
| Secondary: Whisper V1 vs legacy R, seeds 4-10 | best/ | 8.78 | 7.43 | +1.35 [-0.25, +3.02] | 0.29 |
| | final epoch | 10.70 | 8.07 | +2.63 [+0.43, +5.23] | 0.047 |
| Exploratory: Whisper Vall_lt vs R_lt, seeds 4-6 | best/ | 12.41 | 11.06 | +1.35 | 0.59 |
| Nemotron V1 vs R (noam), seeds 1-3 | best.nemo | 9.57 | 10.61 | -1.05 [-2.89, +0.61] | 0.26 |
| | final epoch | 11.51 | 11.21 | +0.30 [-2.22, +2.99] | 0.49 |

The primary test fails both pre-registered criteria: re-voiced copies have no detectable effect. The two Whisper
recipes disagree in sign at best/, and the only interval that excludes 0 (legacy recipe, final epoch) points to harm,
uncorrected for the two recipes. Nemotron agrees with the null. Re-voicing the training clips themselves adds nothing
measurable over the 0.5x synthetic-sentence augmentation above.
