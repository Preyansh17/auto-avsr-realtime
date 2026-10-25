# Higgs Audio v2 LoRA finetune on patient bingge (TTS only), 2026-10-06

**Question.** Does LoRA-finetuning Higgs Audio v2 (3B) on the patient's own ~22 min of speech, with the same in-context
reference prompt as zero-shot, make synthetic speech match the real patient on held-out sentences better than the
zero-shot stitch10 / lowpitch5 / stitch5 / median1 baselines?

**Answer.** Partly. By the pre-registered rule all three finetuned runs **FAIL**, each only on C1b (seed-to-seed PER
spread must be smaller than the gap to the best baseline). Every other criterion passes with margin. The finetuned
runs clearly beat zero-shot on voice match, length control and rambling, and their mean PER and ArtP sit on the real
value. On the 40-sentence test set they did not predict *which sentences* are hard any better than zero-shot stitch5, and
seed noise in PER was as large as the improvement. A 5-fold cross-fitted follow-up on 216 held-out sentences resolves this:
A and B pass all four follow-up criteria (C fails only the ArtP bias), and the finetune beats both zero-shot references on
per-sentence PER error (see Follow-up K). The 3-seed pre-registered verdict stays FAIL.

Code and raw results: `tts_finetune/` (README, scripts, `results/`), cluster mirror `/scratch/pa2753/higgs_ft/`.

## Setup

- Data: patient clips only. `tts_train` 283 clips / 248 sentences / 22.5 min, `tts_val` 22 clips / 20 sentences,
  `tts_test` 49 clips / 40 sentences (split seed 20261005, hash `9341ad8cb2cb`); 40 ASR-test clips excluded.
- Training example = the inference prompt (reference clips of other sentences, 1-5 stitched from `tts_train`) plus the
  target clip's audio tokens; loss on target audio tokens only, EOS label kept.
- LoRA on attention, text MLP and the audio MLP of the DualFFN (280 modules). A: r16, alpha 32, lr 1e-4. B: r64,
  alpha 128. C: r16, lr 5e-5. 15 epochs each, adapter saved every epoch, seed 0. A solo 34 min on one L40S; B and C
  packed as two processes on one L40S took 34 min for both (each ~2.07 min/epoch, same speed as solo).
- Evaluation reference for everything: `ft_stitch5` (5 `tts_train` clips). Zero-shot baselines are the existing
  full-scale runs restricted to the same sentences.

## Phase 0 gates and the strict audit

Gates 1, 3, 4 passed. Gate 2's literal criterion (matched CE clearly below shuffled-audio CE) **failed** (4.417 vs
4.463): same-speaker audio is as predictable under teacher forcing. It was accepted on exact parity with the stock
processor/HF loss and on +-1-frame label shifts (CE 7.0 / 7.6 vs 4.42). A separate stricter audit
(`tts_finetune/strict_gate2/REPORT.md`) then passed T1 (training-path vs inference-path logit parity, 86x noise
margin on shift controls), T2 (overfit 4 clips, free-run reproduces them; both off-by-one controls fail at inference)
and T4 (frame-level audit); T3 failed only its own criterion (other-speaker target audio is undetectable under teacher
forcing). Reference and text conditioning were clearly used. No alignment bug found.

## Phase 3: epoch choice on `tts_val` (20 sentences, 1 seed; noise ~+-0.06 PER)

| run | chosen epoch | PER (real 0.657) | ArtP (0.182) | TitaNet | within +-20% | ramble |
|---|---|---|---|---|---|---|
| pretrained, same ref | n/a | 0.825 | 0.181 | 0.634 | 40% | 0% |
| A_r16 | 12 | 0.675 | 0.136 | 0.700 | 50% | 0% |
| B_r64 | 8 | 0.665 | 0.165 | 0.701 | 65% | 0% |
| C_r16_lr5e-5 | 12 | 0.636 | 0.165 | 0.662 | 70% | 0% |

Teacher-forced validation loss does not track generation: B ended at val loss 5.46 (pretrained 4.39, train 1.73) yet
produced the best audio. A and C collapsed at epoch 1 (length 0.05-0.09x real). Per-run tables: `results/select/`.

## Phase 4: `tts_test` (40 sentences, 49 real clips), 3 generation seeds

| row | PER (real 0.719) | ArtP (0.183) | TitaNet (ceiling 0.690) | WavLM | within +-20% | ramble (>2x) |
|---|---|---|---|---|---|---|
| FT A_r16 e12 | 0.733 +- 0.032 | 0.156 | 0.676 | 0.908 | 76% | 0% |
| FT B_r64 e8 | 0.715 +- 0.048 | 0.164 | 0.670 | 0.898 | 70% | 0% |
| FT C_r16_lr5e-5 e12 | 0.728 +- 0.085 | 0.157 | 0.681 | 0.911 | 77% | 0% |
| zero-shot stitch5 (3 seeds) | 0.781 +- 0.027 | 0.176 | 0.643 | 0.903 | 40% | 8% |
| zero-shot stitch10 | 0.803 | 0.176 | 0.650 | 0.892 | 48% | 10% |
| zero-shot lowpitch5 | 0.455 | 0.331 | 0.594 | 0.888 | 45% | 10% |
| zero-shot median1 | 0.475 | 0.386 | 0.514 | 0.836 | 42% | 8% |

Pre-registered rule (readings frozen in the README before any test generation): C1a PER gap <= 0.07 PASS for all
(0.028 / 0.036 / 0.067); **C1b FAIL for all** (seed range 0.062 / 0.095 / 0.162 vs gap to best baseline 0.048 / 0.067 /
0.053; with sd instead of range A and B would pass, C would not); C2 voice >= 0.64 PASS; C3 length >= 55% and ramble
below every baseline PASS (baselines ramble 5-12.5%); C4 ArtP within 0.04 PASS. The verdict is FAIL as pre-registered.
Three runs share one test set, so even a pass by one would carry a multiple-comparison caveat; the test does not
separate A, B and C, and none is called the headline run.

### Post-hoc (not pre-registered): per-sentence paired analysis (`results/final/post_hoc_paired_2026-10-06.md`)

- PER, mean bias vs real: FT +0.014 / -0.005 / +0.009, stitch5 +0.062. Per-sentence mean absolute error is the same
  (FT 0.205-0.235 vs stitch5 0.215); paired CIs against stitch5 all include zero. So the finetune removes stitch5's
  "reads too impaired" offset but does not track per-sentence difficulty better.
- ArtP, per-sentence error is lower than stitch5 for all three runs (paired -0.026 to -0.039, CIs exclude zero) although the
  mean is slightly further below real (-0.02 to -0.03 vs -0.008).
- Near-duplicate test sentences (15 of 40 within one word of a train sentence): FT PER error is *larger* there
  (0.25-0.39) than on the 25 far sentences (0.14-0.18, FT closer to real than stitch5's 0.217). Memorisation is not what
  is helping. Subsets are small and have no CIs.

## Follow-up S: 10 seeds on `tts_test` (`results/exp_S/table.md`; criteria E1-E4 pre-registered in the README)

All three runs FAIL, only on E2 (is the finetune closer to real PER than zero-shot stitch5, paired over sentences): the
95% CIs of |bias| FT - Z1 are A -0.040 [-0.075, +0.054], B +0.016 [-0.097, +0.116], C -0.027 [-0.087, +0.065]. E1, E3, E4 pass.
Mean PER A 0.726 / B 0.682 / C 0.741 vs real 0.719 (Z2 = pretrained + same `ft_stitch5` reference 0.738, Z1 = stitch5 0.775).
Zero-shot's PER offset is smaller over 10 seeds (+0.042) than over 3 (+0.062), so the 3-seed gap to stitch5 overstated the
benefit. Per-sentence PER error is unchanged (MAE 0.173-0.197 vs 0.180). What holds up with 10 seeds: voice match (0.674-0.680
vs 0.604 Z2 / 0.649 Z1), length control (70-75% within +-20% vs 39-46%), ramble (0-0.2% vs 6.7-10.5%) and per-sentence ArtP
error (0.075-0.080 vs 0.099-0.107). Seed sd of PER is 0.03-0.05, below the sentence-to-sentence spread, so 37 sentences, not
seeds, limit E2; experiment K (229 sentences) is the test of that. Near-duplicate sentences (n=13) have larger FT error
(0.23-0.27) than far ones (n=24, 0.14-0.16).

## Follow-up K: 5-fold cross-fitting, 216 held-out sentences (`results/exp_K/table.md`)

Real patient speech exists only for the 354 non-ASR clips, so the larger held-out set needed retraining: 15 adapters
(A/B/C x 5 folds, same hyperparameters, epochs fixed at 12 / 8 / 12, no per-fold selection), each fold held out by sentence
and generated with `ft_stitch5` (its 5 source clips are in every training set and no held-out fold), 3 seeds. Pool 229
sentences (253 minus 4 `ft_stitch5` sources minus 20 `tts_val`); 216 have PER and ArtP in every row. 40 of the 229 are the
primary `tts_test` sentences; the novel-subset block drops them (178 left with complete data).

**A PASS, B PASS, C FAIL (E4 only).** E2 (closer to real PER than zero-shot stitch5, paired over sentences) passes for all three:
A -0.091 [-0.125, -0.058], B -0.104 [-0.136, -0.065], C -0.062 [-0.098, -0.030].

| row | PER (real 0.679) | bias | ArtP (0.205) | TitaNet | within +-20% | ramble | PER MAE |
|---|---|---|---|---|---|---|---|
| A_r16 e12 | 0.710 +- 0.012 | +0.028 | 0.165 | 0.681 | 75% | 0.1% | 0.185 |
| B_r64 e8 | 0.695 +- 0.020 | +0.015 | 0.177 | 0.670 | 73% | 0.0% | 0.179 |
| C_r16_lr5e-5 e12 | 0.737 +- 0.016 | +0.056 | 0.158 | 0.687 | 72% | 0.1% | 0.174 |
| Z2 (pretrained + ft_stitch5) | 0.741 +- 0.020 | +0.065 | 0.198 | 0.611 | 46% | 6.8% | 0.225 |
| Z1 (stitch5) | 0.797 +- 0.033 | +0.119 | 0.184 | 0.650 | 42% | 13.7% | 0.236 |

- With 216 sentences the finetune does track per-sentence PER better than zero-shot: MAE minus Z1 is -0.050 / -0.057 /
  -0.062 and minus Z2 -0.039 / -0.046 / -0.051, all CIs exclude 0. At n=37 (S, and the post-hoc analysis of the 3-seed run) the
  same difference was unresolvable, so the earlier "per-sentence error unchanged" statement was a power limit, not evidence of
  no effect.
- Not memorisation: near sentences (n=60) are harder for every condition (FT MAE 0.19-0.21, Z1 0.264), and on the 156 far
  sentences FT MAE is 0.159-0.176 vs Z1 0.225 / Z2 0.218.
- E4 is the fragile criterion: A passes the full set at 0.039 (limit 0.04) and fails the novel subset at 0.041; C fails both
  (0.045, 0.048). Only B passes E1, E2, E4 on both. All FT runs sit below real ArtP (-0.027 to -0.045); Z2 is at -0.007.
- PER bias is small and positive (+0.015 to +0.056), versus -0.058 for B on the 37-sentence `tts_test`; per-fold bias ranges
  from -0.050 to +0.101, so single 40-sentence test sets swing by about +-0.06.
- Selection caveat: three runs share one pool, and B being best here was not pre-specified. A/B/C are not separated from each
  other by this experiment (no A-vs-B test run); the supported claim is "finetuning beats zero-shot", not "B beats A".
- The primary 3-seed rule remains FAIL on C1b; K and S are the secondary, pre-registered follow-ups.

## Follow-up R: does the result depend on the reference clips? (`results/exp_R/table.md`)

Until now every finetuned number used `ft_stitch5`. Experiment R regenerated the K held-out set (existing fold adapters, no retraining,
3 seeds) with four 1-5 clip references: `ft_stitch5`, the original `stitch5`, `lowpitch5` and `median1` (one clip), each with a
pretrained model given the same reference as the control. All reference sources are outside the held-out pool and inside every training
set. Criteria R1 and R2 were written into the README before generation.

| reference | finetuned PER (A / B / C) | same-reference zero-shot PER | zero-shot ArtP |
|---|---|---|---|
| ft_stitch5 | 0.710 / 0.695 / 0.737 | 0.741 | 0.198 |
| stitch5 | 0.722 / 0.706 / 0.726 | 0.797 | 0.184 |
| lowpitch5 | 0.697 / 0.677 / 0.731 | 0.581 | 0.326 |
| median1 | 0.733 / 0.713 / 0.747 | 0.498 | 0.387 |

Real PER is 0.679. **R2 passes for all three runs:** the PER range across references is 0.027-0.037 for the finetuned runs against 0.314 for
zero-shot (difference about -0.28, 95% interval [-0.33, -0.22]). **R1 passes for A and B on every reference; C fails once** (PER bias +0.072 on
`median1`, limit 0.07). Voice match is 0.67-0.69 and 70-77% of clips are within +-20% of real length on every reference, against 0.53-0.65 and
42-46% for zero-shot. Per-sentence PER error is lower than the same-reference zero-shot in all 12 run-reference pairs (intervals exclude 0).

Reading: the finetune removes most of the dependence on which clips are in the prompt. Zero-shot swings between reading too clean
(lowpitch5, median1: PER 0.50-0.58, ArtP 0.33-0.39) and too impaired (stitch5: 0.80). The finetuned runs end up slightly more impaired than real
on every reference (bias +0.00 to +0.07) and slightly less articulate by ArtP (0.149-0.178 vs real 0.205). Caveats: 189 of 229 sentences have
PER and ArtP in every row (K had 216); the same three runs and one pool are used throughout; all references are 1-5 clips (stitch10, out of
the training range, was not tested). The test says the output no longer follows the prompt's own impairment level, which is what makes
reference choice a non-issue, but it also means the prompt cannot be used to dial the severity up or down.

## New-text probe (50 sentences, 79% unseen words, 1 seed, no real speech; `results/oov_probe/`)

All three PASS the pre-registered limits. Ramble 0/50 vs zero-shot 7/50 (longest clip 35 s zero-shot vs 10-11 s FT);
short+silent 2 / 0 / 1 of 50; mean PER vs target text 0.706 / 0.638 / 0.710 vs zero-shot 0.703 (paired delta +0.002 /
-0.065 / +0.007, all CIs include zero). No sign of memorised-vocabulary collapse. The probe can fail a model but not
certify that new-text speech sounds like the patient. On the 20 `tts_val` sentences the pretrained model with
`ft_stitch5` did not ramble, so the ramble benefit shows only on new text and on `tts_test` against stitch5/stitch10.

## Caveats

- PER spread across seeds is large on 40 sentences (range 0.06-0.16); a different test set or more seeds could flip C1b.
- stitch10 is out of distribution for the adapter (training used 1-5 clip references); the fair comparator is
  zero-shot with `ft_stitch5` (`tts_val`: PER 0.825).
- `tts_val` / `tts_test` are small; ASR validation sentences (37 of 40) sit inside `tts_train`. Exclude ASR val+test text
  when building prompts for the later ASR-augmentation plan.
- Text-WER / PER of generated audio cannot catch an off-by-one label bug (strict audit, T2); alignment rests on the
  logit-parity test, not on these scores.
- DArtP was not run (reference-free, lowest weight, not part of the rule).

## Still open

- Blind listening set (B_r64 epoch 8, seed 1, 4 sentences x real / finetuned / zero-shot lowpitch5):
  `results/higgs_ft_listening/B_r64_e08/index.html`; answers in the cluster key file, not in that folder.
- Decide whether to proceed to ASR augmentation. The 10-seed and larger held-out-set runs (S, K) are done. Candidate run:
  B_r64 e8 (the only run passing E1/E2/E4 on both the full and novel sets; picked after seeing K). No artifact update has been made.

## Reproduce

`prep_data.py split|tokenize|refs`; `train.sbatch` (A) and `train_packed.sbatch` (B + C); `select_epoch.py plan|score|select`;
`final_eval.py plan|items|table`; `oov_probe.py build|plan|score`; `make_listening_ft.py build`; `post_hoc_paired.py`.
Follow-ups: S gen 19308863-66/19308869, table 19310808; K train 19309298-307 (+ hedges 19310866/71), gen 19311949/53/55/61, items 19311963, PathBench 19311965, neural 19311967, table 19311969 (`cv_prep.py`, `cv_launch_eval.sh`, `multi_eval.py`).
Jobs: A 19260699; B+C 19267405; Phase 3 gen 19262491/92, 19273923/24; Phase 4 gen 19296263-65; PathBench 19302639;
v2 scoring 19302590-19302637; probe 19296290/91/330 + scoring 19302651/54/55; table 19308479.
