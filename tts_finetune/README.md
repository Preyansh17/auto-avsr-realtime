# Higgs Audio v2 LoRA finetune on patient bingge (TTS only)

LoRA-finetune Higgs Audio v2 (3B) on the patient's own ~22 min so that, **with the same in-context reference
prompt format as zero-shot**, synthetic speech matches the real patient on PathBench PER/ArtP, voice and
duration on held-out sentences. Scope: TTS finetune + TTS eval only; ASR augmentation is a later plan.
Cluster copy: `/scratch/pa2753/higgs_ft/` (this directory is the git mirror). Outputs of generation go to
`/scratch/pa2753/gen_runs/FT_<tag>_<ref>_<split>_s<seed>/` in the layout `higgs_gen/gen_higgs.py` writes, so the
existing v2 evaluation (`RUNS_ROOT`) and PathBench scripts work unchanged.

## Pre-registered success rule (written before any FT result was seen)

Evaluated on `tts_test` (40 held-out sentences, 49 real clips), averaged over 3 generation seeds
(seed base 260619825, +1, +2), against the zero-shot baselines restricted to the same sentences:

1. PER gap |synth − real| ≤ 0.07 (the full-scale gap of lowpitch5), and the seed-to-seed spread is smaller than
   the gap to the best baseline.
2. Voice match (TitaNet) ≥ 0.64 (stitch10 level).
3. ≥ 55 % of clips within ±20 % of the real clip's length, and ramble rate (> 2× real length) below every
   zero-shot baseline.
4. ArtP within ±0.04 of real.

**Frozen readings (2026-10-06, written before any `tts_test` generation; implemented in `final_eval.py` / `oov_probe.py`):**
- PER gap = mean over the 3 seeds of the per-seed |PER_synth - PER_real|; seed spread = max - min of the per-seed PER;
  "gap to the best baseline" = |FT mean PER - PER of the zero-shot condition closest to real| (stitch5 = its 3-seed mean).
- Ramble criterion: FT mean ramble rate strictly below each of the six zero-shot baseline rows (stitch5 s1-s3,
  stitch10, lowpitch5, median1); the non-strict reading is printed alongside, not used for the verdict.
- Checkpoint per run = the Phase 3 rule's pick (A_r16 epoch 12, B_r64 epoch 8, C_r16_lr5e-5 epoch 12), no overrides.
  Three runs are tested on one test set: a pass by any run carries a multiple-comparison caveat, and no run is called
  the headline run on test results alone.
- OOV probe verdict limits: FT ramble rate at most 0.04 above zero-shot, SHORT + SILENT clips at most 0.04, mean paired
  PER(FT) - PER(zero-shot) on the new text at most 0.15 (`MAX_RAMBLE_EXCESS`, `MAX_COLLAPSE`, `MAX_PER_DELTA`).

Headline metrics: PathBench PER (primary), ArtP, voice match (TitaNet + WavLM), duration ratio. WER is not a
headline metric (user decision 2026-09-28); DArtP is reported but weighted lowest (LM-hallucination artefact).
Checkpoints are selected on these generation metrics on `tts_val`, **never on train/val loss**.

Phase 3 selection rule: lowest |PER_synth − PER_real| subject to ramble rate ≤ zero-shot stitch10's (same
sentences) and TitaNet voice ≥ 0.60. `tts_val` has 20 sentences, so PER differences below ~0.05 are noise.

## Layout

| file | role |
|---|---|
| `higgs_ft_lib.py` | example construction (prompt = inference prompt + target audio tokens), labels, loss, ref pool |
| `prep_data.py` | `split` (leakage guard, held-out sentences), `tokenize` (audio codes cache), `refs` (inference refs from tts_train only) |
| `phase0_gates.py`, `gates.sbatch` | feasibility gates (forward/loss, teacher-forcing sanity, tokenizer round trip, peft) |
| `train_lora.py`, `train.sbatch` | plain PyTorch LoRA loop, adapter saved every epoch |
| `gen_ft.py`, `build_plan_ft.py` | generation with per-unit adapters; plan/run-folder builder |
| `select_epoch.py`, `final_eval.py`, `ft_metrics.py` | Phase 3 selection, Phase 4 final table + success-rule verdict |
| `oov_probe.py`, `make_listening_ft.py` | generalisation probe and blind listening set |

## Rules that apply to every job here

- Never write under `/scratch/th3482` or `/home/th3482`; all outputs under `/scratch/pa2753/`. `data_june2026_*` is off limits.
- Compute nodes have no internet: `HF_HOME=/scratch/pa2753/hf_cache`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
  Python is `/scratch/pa2753/envs/nemo_asr/bin/python` with `PYTHONPATH=/scratch/pa2753/higgs_site`.
- No heavy work on the login node. Use the GPU as much as possible; pick the account/partition that starts
  soonest across the five SLURM accounts and spread parallel jobs between them (note: `sbatch --test-only`
  start estimates were far too pessimistic on 2026-10-05; jobs started within a minute).
- Final comparisons need ≥ 3 generation seeds (a seed re-roll moved word-error bias by 0.07).

## Known caveats (from the pre-training review, 2026-10-05)

- Gate 2: the plan's "matched CE clearly < shuffled-audio CE" fails (4.417 vs 4.463) because same-speaker shuffled
  audio is as predictable under teacher forcing; alignment is shown instead by exact processor/HF-loss parity and by
  shift +-1 controls (7.0/7.6 vs 4.42). Text conditioning is detectable (wrong text raises CE in 98% of clips).
- Training references use 1-5 clips (<= 45 s). The zero-shot stitch10 reference (10 clips, 51.6 s) is out of
  distribution for the adapter: primary evaluation reference is `ft_stitch5` (also lowpitch5, 5 clips).
- Held-out sentences are exact-match separated from training only: 15/40 test and 3/20 val sentences are within
  one word of a train sentence. The OOV probe, not tts_test, measures true novelty; report near/far separately.
- ASR *validation* sentences (37 of 40) sit inside tts_train; only ASR *test* is guarded (as the plan states).
  Exclude ASR val+test text when building the later ASR-augmentation prompts.
- tts_val has 20 sentences: PER differences below ~0.05 are noise. Zero-shot stitch10 ramble rate on tts_val is
  40 % (8/20), so the Phase 3 ramble constraint is loose; on tts_test the baselines ramble 5-12.5 %, so the ramble
  criterion (below every baseline) needs the FT 3-seed mean < 5 % (2.5 % per clip).
- Real PER/ArtP in the final table are means over all real repeats of a sentence (not the first clip per sentence
  used by the older 0.665 figure), so numbers differ slightly from earlier tables.

## Follow-up experiments (pre-registered 2026-10-06, after the 3-seed rule FAILED on C1b, before any new generation)

The 3-seed verdict above stays the primary pre-registered result. These two experiments measure what C1b was trying to
detect (is the improvement over zero-shot larger than seed/sentence noise?) with less noise. Both are secondary.

- **S (more seeds):** tts_test (40 sentences), the same checkpoints (A_r16 e12, B_r64 e8, C_r16_lr5e-5 e12), seeds 1-10,
  plus zero-shot controls with identical seeds: Z2 = pretrained + `ft_stitch5` (same reference as the finetuned runs) and
  Z1 = zero-shot stitch5 (original reference, seeds 4-10 generated now; seeds 1-3 are the existing baseline runs).
- **K (larger held-out set, 5-fold cross-fitting):** real patient speech exists only for the 354 non-ASR clips, so a
  larger held-out set needs retraining. Pool = every sentence that is not ASR-test, was generated by the zero-shot runs
  (paired baselines exist) and is not the source sentence of ANY inference reference: the 253-sentence pool minus the 4
  sentences that are sources of the `ft_stitch5` reference = 249 (an independent pre-training review found that the first
  fold draft held those 4 out, which would have put a held-out sentence's own audio and text in the prompt; folds were
  rebuilt, nothing had trained), minus the 20 tts_val sentences (kept in training; they chose the epochs) = **229
  sentences**, split into 5 word-count-stratified folds (seed 20261007) by sentence (all repeats together; 46/46/46/46/45).
  For each fold, A/B/C are retrained on every other non-ASR clip (the same hyperparameters, seed 0, 15-epoch cosine
  schedule, stopped after the already chosen epoch 12 / 8 / 12; no per-fold selection) and generate that fold's sentences
  with `ft_stitch5` (all 5 of its sources are in every training set and in no held-out fold), 3 seeds. Controls: Z2 on the
  same sentences/seeds; Z1 and stitch10/lowpitch5/median1 = the existing full-scale baseline runs restricted to the same
  sentences. Held-out evaluation pools all 229 sentences (out-of-fold). **Disclosure:** 40 of the 229 are the primary
  tts_test sentences that were already scored; the pooled verdict uses all 229, and E1, E2, E4 (the PER/ArtP criteria) are
  also reported on the 189 sentences that were never in the primary test.
- **Criteria (per finetuned run X, over all seeds and sentences S of the experiment; d_i = mean-over-seeds synthetic PER
  of sentence i minus mean real PER of sentence i):**
  E1 `|mean_i d_i| <= 0.07`;
  E2 paired bootstrap over sentences (10,000 resamples, seed 0): the 95% CI of `|mean d_X| - |mean d_Z1|` lies entirely
  below 0 (X is closer to real than zero-shot stitch5); the same quantity against Z2 is reported as secondary, not
  part of the verdict;
  E3 mean TitaNet >= 0.64, share within +-20% of real length >= 55%, and mean ramble rate strictly below each zero-shot
  condition's mean (Z1, Z2, and the single-seed stitch10 / lowpitch5 / median1);
  E4 `|mean ArtP_synth - ArtP_real| <= 0.04`.
  Verdict PASS only if E1-E4 all hold. Seed spread (sd over seeds) is reported, not a criterion. Near-duplicate sentences
  (word edit distance <= 1 to any training sentence of that run) are reported separately from far ones. Reporting also
  includes per-sentence mean absolute error and its paired CI versus Z1 and Z2.

## Experiment R: reference sensitivity (pre-registered 2026-10-07, before any generation with a new reference)

Question: do the finetuned results depend on the 5 clips used as the prompt? Every finetuned number so far used `ft_stitch5`.
Same cross-fitted held-out set as K (229 sentences, 5 folds, the 15 existing fold adapters at epochs 12 / 8 / 12, seeds 1-3, no
retraining), generated with four reference prompts: `ft_stitch5` (the K runs, reused), `stitch5` (original 5-clip stitch),
`lowpitch5` (5 deliberately low-pitched clips) and `median1` (one clip, 5.8 s). All are 1-5 clips and under 45 s, so inside the
training prompt distribution; `stitch10` is left out for the earlier out-of-distribution reason. Every source clip of every one
of these references is excluded from the held-out pool by `cv_prep.py` and is in every training set (checked by its asserts), so
no reference contains a held-out sentence. Control Z(r) = pretrained model with the same reference r, same code path and seeds.
Code: `ref_eval.py` (`plans`, `items`, `table`), launched by `ref_launch.sh`.

- **R1 (quality holds for each reference), per finetuned run X and reference r:** |mean PER - real| <= 0.07, TitaNet >= 0.64,
  >= 55% of clips within +-20% of real length, and ramble rate strictly below Z(r)'s. Run verdict PASS only if all four
  references pass. ArtP is reported, not part of the verdict (it was the fragile criterion in K).
- **R2 (finetuning reduces reference sensitivity), per run:** the range (max - min) over the four references of the mean PER of
  X minus the same range for Z; paired bootstrap over sentences (10,000 resamples, seed 0): the 95% CI of the difference lies
  entirely below 0.
- Reported, not in a verdict: per-reference |bias| and per-sentence MAE of X minus Z(r) with paired CIs; spread of ArtP, TitaNet
  and within-+-20% across references.
- Sentence set: the intersection with PER and ArtP in every row (as K). PathBench scores for `ft_stitch5` runs and the real clips are
  reused from `exp_K`. Three runs share one pool (multiple-comparison caveat as before).

## Experiment RS: random-reference study (fold 0; registered 2026-10-07 before generation; `ref_study.py`, `refstudy_launch.sh`)

Complements experiment R (above, run in a parallel session) with references drawn at random. How much does the inference
  reference matter? Fold 0 of K (46 held-out sentences), seeds 1-3, adapter B_r64_cv0 (epoch 8) and the pretrained model, each
  with 8 references built only from fold-0 training clips: `ft_stitch5` (existing K runs), original `stitch5`, `lowpitch5`,
  one random clip (`rs_rand1`), three random clips (`rs_rand3`), two random 5-clip draws (`rs_rand5a/b`, sentence-disjoint) and a
  greedy vocabulary-diverse 5-clip stitch (`rs_vocab5`). Reported per reference: PER/ArtP bias, per-sentence MAE with paired
  bootstrap vs `ft_stitch5`, voice, length, ramble, plus the spread across references for B vs zero-shot (does finetuning reduce
  reference sensitivity?) and the rand5a-vs-rand5b draw noise. One fold, 46 sentences: a winner must be confirmed on a second fold
  before it is adopted. Exploratory, no pass/fail criteria.

**RS result (`results/exp_RS/table.md`; fold 0, 36 sentences with full data, B_r64_cv0 vs pretrained, 3 seeds):** no random or
vocabulary-diverse reference beats `ft_stitch5` (all paired MAE CIs include 0). Across the 8 references finetuned B's PER bias
ranges 0.080 (zero-shot 0.220), TitaNet 0.660-0.680 (zero-shot 0.479-0.642), within +-20% 65-78% (zero-shot 35-54%), ramble
<= 0.7% (zero-shot 3.6-13.0%). A single random 7.6 s clip still gives TitaNet 0.670 for B, 0.479 zero-shot. Draw noise: rand5a vs
rand5b differ by 0.037 PER bias (seed sd 0.021). Vocabulary-diverse selection did not help. Conclusion: with the finetune the
reference is not worth optimising; rotate references for diversity. One fold, one adapter, 36 sentences.

## Results (2026-10-06)

Full write-up: `docs/investigations/higgs_lora_finetune_2026-10-06.md`; raw tables in `results/`.

- Phase 0: gates 1, 3, 4 pass; gate 2 accepted on parity + shift evidence, then confirmed by the stricter audit
  (`strict_gate2/REPORT.md`: T1, T2, T4 pass; T3 fails only its own criterion).
- Phase 2/3: A_r16 (epoch 12), B_r64 (epoch 8), C_r16_lr5e-5 (epoch 12) chosen on `tts_val` generation metrics
  (`results/select/`); validation loss does not track generation quality.
- Phase 4 (`tts_test`, 3 seeds): **all three runs FAIL the pre-registered rule, only on C1b** (seed spread vs gap to the best
  baseline); C1a, C2, C3, C4 pass. FT mean PER 0.715-0.733 vs real 0.719 (stitch5 0.781); TitaNet 0.670-0.681 (real ceiling
  0.690, stitch10 0.650); 70-77% of clips within +-20% length (zero-shot 40-48%); ramble 0% (zero-shot 5-12.5%).
  `results/final/final_table.md`, `success_rule.json`.
- Post-hoc paired per-sentence analysis (not pre-registered): mean PER bias removed, per-sentence PER error unchanged
  vs stitch5, per-sentence ArtP error lower. `results/final/post_hoc_paired_2026-10-06.md`.
- New-text probe (50 sentences): all three pass; ramble 0/50 vs zero-shot 7/50. `results/oov_probe/`.
- **Experiment S (10 seeds, `tts_test`; `results/exp_S/table.md`):** A/B/C all FAIL, only on E2 (paired CI of |bias| FT - Z1
  includes 0: A -0.040 [-0.075, +0.054], B +0.016 [-0.097, +0.116], C -0.027 [-0.087, +0.065]); E1, E3, E4 pass. Mean PER
  A 0.726 / B 0.682 / C 0.741 vs real 0.719, Z2 0.738, Z1 0.775 (Z1's 3-seed bias of +0.062 was partly seed luck: +0.042 over
  10). Seed sd of PER 0.03-0.05. Per-sentence PER MAE 0.173-0.197 vs Z1 0.180 (no difference). Clear wins over zero-shot are
  voice (TitaNet 0.674-0.680 vs Z2 0.604 / Z1 0.649), length (70-75% within +-20% vs 39-46%), ramble (0-0.2% vs 6.7-10.5%) and
  per-sentence ArtP error (0.075-0.080 vs 0.099-0.107). 37 of 40 sentences have PER and ArtP in every row.
- **Experiment K (5-fold cross-fitting, 229-sentence pool, 216 with complete PER+ArtP data in every row; 3 seeds;
  `results/exp_K/table.md`):** **A PASS, B PASS, C FAIL (E4 only: |ArtP bias| 0.045 vs 0.04).** E2 passes for all three: |bias|
  FT - Z1 = A -0.091 [-0.125, -0.058], B -0.104 [-0.136, -0.065], C -0.062 [-0.098, -0.030]. PER A 0.710 / B 0.695 /
  C 0.737 vs real 0.679 (bias +0.028 / +0.015 / +0.056), Z2 0.741, Z1 0.797. Per-sentence PER MAE 0.174-0.185 vs Z1 0.236 and
  Z2 0.225; paired CIs vs both exclude 0 for all three (e.g. B vs Z2 -0.046 [-0.069, -0.024]). This reverses the n=37 picture:
  with 216 sentences the finetune does track per-sentence PER better than zero-shot. TitaNet 0.670-0.687, 72-75% within
  +-20%, ramble 0-0.1% (zero-shot 6.8-13.7%). The advantage holds on the 156 far sentences (FT MAE 0.159-0.176 vs Z1 0.225 /
  Z2 0.218); near sentences are harder for every condition. Novel subset (178 sentences never in the primary `tts_test`): E1 and
  E2 pass for all; E4 fails for A (0.041) and C (0.048), passes for B (0.032). A's E4 pass on the full set (0.039) is within
  0.001 of the limit. Fold bias is mixed (fold 0 negative for A/B, folds 3-4 positive for all), seed sd 0.012-0.020.
  Vs Z2 (same reference), A and B are clearly closer in bias (CIs exclude 0), C is not. Secondary result; the 3-seed pre-registered
  rule above stays FAIL on C1b. A/B/C share one pool, so a pass by any run carries the multiple-comparison caveat.
- **Experiment R (reference sensitivity; 4 references x A/B/C + same-reference zero-shot, K held-out set, 3 seeds; `results/exp_R/table.md`):**
  R1 (quality holds for every reference): **A PASS, B PASS, C FAIL** (C misses PER bias on `median1`: +0.072 vs the 0.07 limit). R2
  (finetuning shrinks the spread of mean PER across references): **PASS for all three**, range 0.027-0.037 for finetuned vs 0.314 for the
  same-reference zero-shot (difference -0.28 [-0.33, -0.22]). Zero-shot PER swings from 0.498 (`median1`) to 0.797 (`stitch5`); finetuned
  stays 0.677-0.747. Per-sentence MAE is lower than same-reference zero-shot for all 12 run-reference pairs (CIs exclude 0). 189 of 229
  sentences have PER and ArtP in every row (K had 216).
- Not done: DArtP; user listening session (`results/higgs_ft_listening/B_r64_e08/`); artifact update (needs user OK).
