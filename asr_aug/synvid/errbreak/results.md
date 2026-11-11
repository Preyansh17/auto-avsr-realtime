# In-home chain error breakdown (0 GPU), 2026-11-11

Data: deployed v1.8 chain (F1 + q8 fusion 0.1 + Qwen2.5-0.5B re-rank, app45) on `semg3_rcv_20261109_v1`, 193 in-home
clips, 1116 words, **81 errors** (rebuilt exactly as `pbias/step0_rescore.py`). Script `errbreak.py` (counts in
`/scratch/pa2753/synvid/errbreak/errbreak.json`), `pct.py` for the rh4sp check. Counts only; no transcripts here.

**Precedence (fixed before counting):** 1 spoken-form > 2 final word > 3 unseen in fold training text > 4 seen-word
substitution > 5 deletion/insertion. "Unseen" = not in any `rcv{k}_*_train` role or `rcv{k}_real_all` of the clip's fold;
"has video" = word occurs in the 917-row real-video pool. 20-best = first 20 distinct chain candidates (the LM's list).

| bucket | errors | clips | ops | token fixed by 20-best oracle | errors in clips whose exact ref is in 20-best |
|---|---|---|---|---|---|
| 1 "percent" / numbers / units | 10 | 10 | 8 del ("percent"), 2 sub | 8 | 8 (8 clips) |
| 2 final word (other) | 12 | 12 | 12 sub | 10 | 8 (8 clips) |
| 3a unseen, no real video | 11 | 9 | 10 sub, 1 del | 3 | 3 (3 clips) |
| 3b unseen, has real video | 0 | 0 | — | — | — |
| 4 seen-word substitution | 24 | 18 | 24 sub | 13 | 10 (9 clips) |
| 5 deletion / insertion | 24 | 20 | 17 del, 7 ins | 14 | 10 (9 clips) |
| **total** | **81** | 47 with errors | | **48** | **39** (30 of 47 clips) |

- Re-ranking ceiling: a perfect pick from the existing 20-best fixes 39 errors (sentence oracle) to 48 (token oracle).
  The other ~33 need a better visual model (the right answer is not in the list).
- Unseen words: 11 errors, mostly names and rare words; none has real video in the pool, and only 3 are in the 20-best.
  Synthetic or real support video could reach at most these 11 (and A14-A18 showed ~4-5 in practice).

## Tianyu's spoken-form run (rh4sp)

`/scratch/th3482/Dysarthric_ASR/runs/s3v34/submit_tw1.sh` (TW1): per fold k, trained on rcv{k}_s3_train + s3ov_train +
real_all x4 + v34_train + **home_train x4**, speed/trunc/affine/photo aug, `--target-style spoken`, 1,000 updates;
CTC 0.1 (rh4sp) vs 0.3 (rh4spc3). Plain beam on rcv{k}_home_val (not the deployed chain).

| run | in-home errors / 1116 (folds 0-4 summed) |
|---|---|
| F1 plain (reference, pbias) | 112 |
| rh4sp seed 1 / 3 / 7 / 42 | 95 / 102 / 106 / 99 |
| rh4sp 4-seed soup | 100 |
| rh4spc3 seed 1 / 7 | 96 / 108 |
| deployed chain (F1) | 81 |

"Percent" clips: **all 8 exactly correct** in every rh4sp seed, the soup, and soup + fusion (chain gets 0/8).
rh4sp has not been put through the full chain (fusion + Qwen re-rank); that combination is untested.

## Unlabelled in-home clips for self-training

None. All 193 in-home clips (`inhome/`) are labelled and already the CV eval set. The only unlabelled clips in any split
are 66 `real_unl/` phone clips (domain s3real, not in-home), which Tianyu already pseudo-labelled
(`dc_pseudoA_f0-2` 63-64 rows, `dc_pseudoB_f0-2` 27-29 rows). `derived/devcrop_v1` is not readable from pa2753, so
unlisted crops there are unverified.

## Levers ranked by errors reachable

1. **Spoken-form targets + home x4 (rh4sp) through the deployed chain** — ~8-10 errors (bucket 1, shown 8/8 on plain
   beam) plus rh4sp's general plain-beam gain (112 -> 95-106). Cheapest: models exist, only chain decode/re-rank needed.
2. **Better n-best re-ranking** — ceiling 39-48 errors, but past attempts (pbias bias list: 0 on chain; LM variants)
   realised little; worth it only with a new signal (e.g. per-clip audio-free context, home-train phrase priors on rh4sp).
3. **Better visual model / more in-home recordings** — ~33 errors not in the 20-best; needs new data from the patient.
4. **Unseen-word coverage** (support video, vocabulary) — <= 11 errors, ~4-5 realistic (A14-A18); closed.
5. **Self-training** — no unlabelled in-home data exists; 66 phone clips already used.

## Lever 1 tested: rh4sp through the deployed chain (0 GPU), 2026-11-11

Tianyu had already decoded the rh4sp 4-seed soup (`runs/tw1/rh4sp/f{k}/soup4_a1.0/v`, int8+fp16, beam 10, CTC 0.1,
Qwen fusion 0.1, app45) and scored its n-best with Qwen2.5-0.5B. `rh4chain.py` applies the deployed re-rank (identical
to the F1 baseline) and a paired clip bootstrap vs F1 chain (`/scratch/pa2753/synvid/errbreak/rh4chain.json`).

| model through chain | errors / 1116 | vs F1 chain [95% CI] | clips fixed / broken | "percent" clips exact |
|---|---|---|---|---|
| F1 (deployed) | 81 | — | — | 0/8 |
| **rh4sp 4-seed soup (primary)** | **84** | **+3 [-20, +27]** | 18 / 18 | 8/8 |
| 15 other existing Tianyu chain variants (descriptive) | 75-86 (one broken run: 252) | best -6 [-24, +11] | | 3-8/8 |

Reading: spoken-form targets fix all 8 "percent" clips through the chain too, but the model change swaps ~18 clips each
way, so the net is no better than F1. None of the 15 other variants is clearly better either (best -6, CI includes 0,
and picking the best of 15 is selection). The 112 -> ~100 plain-beam gain does not survive the LM re-rank: the chain
was already recovering most of it.

## "percent" post-rule on the deployed chain (0 GPU), 2026-11-11

Rules frozen in `pctrule.py` docstring before scoring. R1 (primary): if the chain output ends in a number word and the
20-best contains exactly output + "PERCENT", use it. R2 (descriptive): if it ends in a number word and any 20-best
candidate contains PERCENT, append PERCENT. Output `/scratch/pa2753/synvid/errbreak/pctrule.json`.

| arm (193 in-home clips) | errors / 1116 | vs chain [95% CI] | fires (on percent refs) | fixed / broken | "percent" exact |
|---|---|---|---|---|---|
| chain (deployed) | 81 | — | — | — | 0/8 |
| R1 (primary) | 80 | -1 [-3, 0] | 1 (1) | 1 / 0 | 1/8 |
| R2 (descriptive) | 74 | -7 [-13, -3] | 7 (7) | 7 / 0 | 7/8 |

Do-no-harm (F1 plain top-1, no app context): R1 and R2 fire 0 times on dc_s3_val 30, dc_s3ov_val 35, dc_v34_val 43,
dc_s3_test 30, dc_s3_oov 50 clips (errors unchanged). Weak check: those sets contain no "percent" and only 2 references
ending in a number, so they barely exercise the rule.

Caveats: the rule was designed after seeing these in-home errors, so -7 is in-sample; R2 was not the primary. The real
test is new home recordings (does any non-percent sentence ending in a number get a wrong PERCENT appended?).

## Lever 3 check: Tianyu's personal-LM re-ranker (read-only, 0 GPU), 2026-11-11

Source: `/scratch/th3482/Dysarthric_ASR/runs/s3v34/home_eval_v1/plm2/tune_rh4sp_{s1,soup71}.json` (home_lm.py tune).
Personal LM = Qwen2.5-0.5B fully fine-tuned per fold on the patient's in-home sentences of the other folds (~154,
3 epochs, lr 1e-5; `code_dfd2f10e/home_plm.py`), re-ranking rh4sp n-best. Weights (lam, mu) picked on training folds
(nested CV, honest). "ref" = deployed App chain top-1 under Tianyu's scorer: 84 errors (81 under ours; normalisation differs).
Paired bootstrap from Tianyu's file.

| VSR model + personal LM (context) | errors / 1116 (CV) | vs App chain [95% CI] |
|---|---|---|
| rh4sp seed 1, app45 (deployed context, pre-specified) | 70 | -14 [-36, +7] |
| rh4sp seed 1, best of 16 contexts (app8_w600) | 69 | -15 [-38, +6] |
| rh4sp soup, app45 | 83 | -1 [-25, +23] |
| rh4sp soup, best of 16 (app8_w600) | 74 | -10 [-35, +14] |
| rh4sp seed 1, no context (c0) | 84 | 0 [-22, +21] |

Reading: no variant is clearly better than the deployed chain (every CI includes 0). The same re-ranker gives 70 on one
seed and 83 on the 4-seed soup, so seed noise (~13 errors) is as big as the effect. App context matters (c0 = no gain).
Best-of-16 numbers are selected. Not worth more 0-GPU re-ranking work on 193 clips; revisit when new home recordings
add both personal-LM training text and a fresh test set.
