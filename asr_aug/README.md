# Higgs synthetic augmentation for patient ASR

Does synthetic patient speech from the finetuned Higgs v2 adapter (B_r64 epoch 8, see
`docs/investigations/higgs_lora_finetune_2026-10-06.md`) improve ASR on the real split_v1 merged test set?
Cluster mirror: `/scratch/pa2753/asr_aug/` (code/, data/, root/, runs/, logs/, samples/, jobs.txt).

## Pipeline

1. `build_sentences.py` — pool A: 1000 new sentences from the 76-word ASR-train vocabulary; pool B: 1000 open-vocab
   everyday/AAC sentences (304 word types, 51% of tokens outside train vocab). Grammar-sampled, seed 20261012.
   Leakage audit (`data/audit.json`): 0 sentences equal or within one word edit of any ASR val/test sentence;
   0 contain a val/test word unseen in train (bad, but, hello, music, tired); pool A has no exact train sentence.
   All 2000 round-trip through SPM-1023 losslessly.
2. `make_gen_plan.py` — one clip per sentence, references rotated over ft_stitch5 / stitch5 / lowpitch5 / median1
   (500 each), adapter `higgs_ft/runs/B_r64/epoch_08`, seed 1. Jobs 19613790/91, 19615575 (8 x ~15 min, L40S).
   Synthetic median 0.86 s/word = real train 0.86.
3. `filter_synth.py collect|score` + `judge.sbatch` (job 19616213). Judge = real-only Whisper large-v3 best recipe,
   seed 1 (val WER 10.7%). Thresholds = 95th percentile on real val: WER 0.576, insertions 1, CER 0.521. Effective rules: pool A rejected if WER > 0.576
   or >= 2 insertions (90% of real val passes this OR of two rules, not 95%); pool B rejected if CER > 0.521.
   Kept: A 894/1000 (dur 23, silent 1, judge 102), B 647/1000 (dur 21, silent 2, judge 350).
   Listening samples per reason: `/scratch/pa2753/asr_aug/samples/{kept,dur,silent,judge}/`.
4. `whisper_aug.sbatch` + `make_mix.py` — large-v3, `slurm/whisper_splitv1_best.sbatch` recipe (full FT + SpecAugment,
   lr 1e-5, warmup 25, wd 0.01, 30 epochs, val-WER checkpoint selection), offline greedy decode of val and test.
   The synthetic subset is fixed per arm x seed (drawn with the seed), not resampled per epoch; 30 epochs over the
   combined set = 30 passes over the real data, so steps grow with the ratio.
5. `report.py` — test WER on all 40 / known-word 36 / unseen-word 4 (sentences containing bad, hello, music, tired),
   seed mean +- sd, paired bootstrap vs R over sentences, unseen-word hit counts.

## Arms (3 seeds each)

| arm | train data |
|---|---|
| R | real only = existing `whisper_largev3_specaug_splitv1_merged_wdwarmup_seed{1,2,3}` (re-decoded: test 7.62 / 6.28 / 8.97%, identical to their logs) |
| S | all kept synthetic (1541), no real |
| M0.5 / M1 / M2 / M4 | 314 real + 157 / 314 / 628 / 1256 synthetic, half pool A, half pool B |
| ablations | at the selected ratio r*: A-only (A r*), B-only (B r*), duration+silence filter only (D r*) |

## Pre-registered decisions (written 2026-10-12, before any augmented test result)

- r* (ratio for ablations and for Nemotron) = the M arm with the lowest **mean val WER** over 3 seeds; ties (within
  0.5 pp) go to the smaller ratio. Test WER is not used to choose. If r* = 4, the B-only ablation runs at r = 2 (pool B
  has 647 kept clips < 1256).
- Augmentation "helps" only if the paired bootstrap 95% interval of test WER(M r*) - WER(R) on all 40 sentences lies
  below 0. A point estimate below R with an interval crossing 0 is reported as "no detectable effect".
- The unseen-word subset (4 sentences) is reported but never used for a claim on its own.
- Ablations and Nemotron are secondary analyses of r*; no further ratios are added after seeing test results.

## Results so far (3 seeds, 2026-10-13)

Whisper test WER (R 7.62): S 12.41 (+4.78 [+1.63, +8.10]); M0.5 7.03 (-0.60 [-3.00, +1.72]); M1 7.47; M2 7.77; M4 7.32;
ablations at r* = 0.5: A0.5 5.83 (-1.79 [-4.19, +0.29]), B0.5 8.37 (+0.75), D0.5 5.68 (-1.94 [-3.89, -0.30]).
Nemotron (R 10.61): M0.5 8.67 (-1.94 [-4.76, +0.61]); S 15.70 (+5.08 [-3.12, +14.14]; S trained 43 epochs = R's step count).
Pre-registered primary (M0.5 vs R): no detectable effect. M0.5 and D0.5 differ in the filter and in an independently drawn synthetic subset (11-16 of 157 clips shared), yet are 1.35 pp apart,
so synthetic-subset + seed noise at n=3 is of the same size as every effect seen; D0.5's interval excluding 0 is one of
8 post-hoc comparisons and is not a finding.

### Audit disclosures (2026-10-13, post hoc)

- **Val is not independent of the synthetic data.** The Higgs split (`tts_finetune/prep_data.py`) drops only ASR-test
  sentences, so 37 of the 40 ASR val clips are in the adapter's tts_train (1 in tts_val, 2 in tts_test). The ft_stitch5
  reference (500 of 2000 generations, 382 kept) also splices in one ASR val clip ("But I do not know it"). Val drives
  checkpoint selection, r* and the judge thresholds, so val WERs of augmented arms are biased low (their val gains over R
  are 3-6x their test gains) and are not comparable with R. Test is clean: no test clip or test sentence reached the TTS,
  its references, or any train CSV/manifest. Every test-based number stands, but best/ checkpoints of augmented arms were
  picked on a val set the generator partly saw.
- 10 of 40 val sentences appear verbatim in real train (split_v1 design; test has none), so val is an easier proxy for test
  in every arm.
- The pool-B judge (R seed 1, a 76-word-vocabulary model) cannot output most pool-B words, so the CER rule partly selects
  pool-B clips by how train-like their text is, not only by audio quality. "Judge filter" contrasts (M vs D) mix the two.
- Thresholds come from n=40 (95th pct = between the 2nd and 3rd largest values), so the kept-set size is sensitive to them.
- Whisper S is epoch-matched (30 epochs over 1541 clips = 2910 steps, 4.85x R); Nemotron S is step-matched to R.
- R seeds 1-3 were trained in July with an older cluster code snapshot and HF cache; their re-decode is word-identical to
  the July logs. The seeds 4-10 R reruns use the same code and environment as the augmented arms.

## Pre-registered confirmatory test: fresh seeds 4-10 (written 2026-10-13, before any seed 4-10 run exists)

- Arms: R, M0.5 (the pre-registered r*), A0.5 and D0.5 (the two best 3-seed ablations, promoted post hoc), seeds 4-10,
  Whisper large-v3 recipe unchanged. Each seed draws its own synthetic subset (`make_mix.py`), so the result averages over
  subsets as well as seeds. R uses the real train CSV unshuffled, as the original R runs did.
- Primary analysis uses seeds 4-10 only (7 fresh seeds per arm). Statistic: test WER(arm) - WER(R), mean over the 7 seeds,
  95% interval from a two-level bootstrap (resample seeds within arm, then sentences, 10000 draws; `report.py --seeds 4-10 --boot2`).
- Decision: an arm "helps" if its interval lies below 0 after Holm correction over the 3 comparisons (family-wise 0.05).
  Otherwise "no detectable effect". All-10-seed numbers are reported as secondary.
- Exploratory only (no claim): M0.25, seeds 1-3.
- Added 2026-10-13, after the audit and before any seed 4-10 run finished (the first launch failed on a full /scratch quota and
  was resubmitted): val-free secondary. Each seed 4-10 run also decodes its final-epoch checkpoint (`{val,test}_hyps_last.tsv`),
  which no val signal chose, because val is not independent of the synthetic data (audit disclosures above). Same two-level
  bootstrap + Holm (`report.py --seeds 4-10 --ckpt last --boot2 M0.5 A0.5 D0.5`). If primary and secondary disagree, report both;
  a "helps" claim needs both.

## Confirmatory result (seeds 4-10, 2026-10-13): no detectable effect

Jobs 19730216-19730497 (first launch 19689031-064 failed: 31 concurrent runs x 52-69 GB checkpoints filled the 5 TB /scratch
quota; relaunched in 8 throttled lanes, `whisper_launch.sh`). Reports: `data/report_confirm_{best,last}.md`, `data/report_all10_best.md`.

| analysis | R test WER | M0.5 | A0.5 | D0.5 |
|---|---|---|---|---|
| primary: val-selected best/, seeds 4-10 | 7.43 | 8.78 (+1.35, p=0.34) | 6.92 (-0.51, p=0.76) | 7.56 (+0.13, p=0.95) |
| secondary: final epoch, seeds 4-10 | 8.07 | 6.47 (-1.60, p=0.32) | 6.34 (-1.73, p=0.29) | 6.79 (-1.28, p=0.41) |
| secondary: best/, all 10 seeds | 7.49 | 8.25 (+0.76, p=0.51) | 6.59 (-0.90, p=0.49) | 7.00 (-0.49, p=0.64) |

Two-level bootstrap + Holm: no arm "helps" in any analysis. The 3-seed leads of A0.5 (-1.79) and D0.5 (-1.94) did not
replicate on fresh seeds (-0.51, +0.13); M0.5 flipped sign. Final-epoch checkpoints point the same way for all three arms
(-1.3 to -1.7 pp) but every interval crosses 0, and R's own final epoch is 0.6 pp worse than its val-selected one while the
augmented arms' final epochs beat their val-selected ones: consistent with val (seen by the TTS) misleading checkpoint
selection for augmented arms. Seed sd of R is ~1.5 pp on 40 sentences, so effects under ~2 pp are not resolvable here.

Post-hoc checks on the confirmatory data (not pre-registered). Seed-level tests on per-seed test WER (7 vs 7):
val-selected best/ M0.5 +1.35 (Welch p=0.20), A0.5 -0.51 (0.61), D0.5 +0.13 (0.89); final epoch M0.5 -1.60 (0.072),
A0.5 -1.73 (0.040, exact permutation 0.057), D0.5 -1.28 (0.20). Nothing survives Holm over 3. The audit's statistics lens
(unverified) argues `--boot2` double-counts the seed x sentence interaction and is conservative, so the null above is
"no effect resolvable here", not evidence of no effect; the seed-level tests give the same answer.
Exploratory M0.25 (seeds 1-3): 9.27 +- 2.70 vs R 7.62 (+1.64 [-2.15, +6.44]). Lower ratios do not help either.

## Nemotron, selection-free check (job 19783348, 2026-10-14)

Audit (confirmed by 2 verifiers): R seeds 2-3 (July) picked nemotron_best.nemo with the pre-76157a3 SaveBestWER (best reset
on every resume -> best of the last link only), so best-vs-best deltas mix selection rules. All 9 runs' final-epoch models
(nemotron_patient.nemo) decoded on val and test (`nemotron_decode_last.sbatch`; `report.py --model nemotron --ckpt last`).

| checkpoint | R | M0.5 | S |
|---|---|---|---|
| best.nemo (mixed rules) | 10.61 (11.21/9.87/10.76) | 8.67 (-1.94 [-4.76, +0.61]; Welch over seeds p=0.020) | 15.70 (+5.08; Welch p=0.001) |
| final epoch (no selection) | 11.21 (11.21/10.76/11.66) | 9.57 (-1.64 [-3.79, +0.57]; Welch p=0.23) | 16.29 (+5.08; Welch p<0.001) |

M0.5's every-seed lead under best.nemo weakens once selection is equalised (seeds 10.31/7.62/10.76): no detectable effect.
Synthetic-only S is reliably worse than R across seeds on this test set (sentence-level interval still crosses 0).
Also fixed for future runs: SaveBestWER saved the epoch after the best (Lightning runs callback epoch-end hooks before NeMo
logs val_wer); now on_validation_end. Effective Nemotron LR is a flat 1e-6 (NoamAnnealing peak 2.2e-7 clamped to min_lr),
not 1e-4, in every run. Nemotron M0.5 gets 1.5x R's optimizer steps (74,340 vs 49,560).
