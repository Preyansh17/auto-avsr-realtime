# pbias results — decode-only phrase bias on semg3 in-home CV (2026-11-10)

Design frozen in `README.md` before scoring. Data: `semg3_rcv_20261109_v1` random 5-fold, 193 home_val clips, 1116 words
(885 seen in the fold's home_train / 231 unseen). Compute: 0 GPU-hours. Every arm re-ranks n-best lists Tianyu had already
decoded; the scripts ran on the torch login node, about 10 s each. Outputs: `/scratch/pa2753/synvid/pbias/{step0,step1}.json`.
No test53/test57/OOV50/in-home test48 decoding.

## Step 0: Tianyu's pvocab

- **Method.** In-beam bonus per sub-word token while a hypothesis spells a listed word. The bonus is taken back if the word is
  not finished (`s3d_bias.py` via `usr2_eval.py --bias-words/--bias-bonus`). It ran on F1 plain beam 10, CTC 0.1 only, not on
  the chain.
- **List.** Built per fold from the **rare** words of that fold's home_train transcripts (words outside a general top-5000
  list), 21–27 words per fold. Every list word occurs in that fold's home_train, so there is **no leakage**. 1–5 list words
  per fold also recur in home_val, which is legitimate. Because of the rarity filter, "percent" itself
  and most number words are **not** listed.
- **Strength.** {1, 2, 3} per token.

| arm (F1 plain, 193 clips) | errors / 1116 | seen err | unseen err | last-word err | Δ vs plain [95 % CI] |
|---|---|---|---|---|---|
| plain (b0) | 112 (10.04 %) | 64 | 32 | 24 | — |
| pvocab b1 (= LOFO pick on every fold) | 109 | 61 | 32 | 24 | −3 [−7, 0] |
| pvocab b2 | 110 | 61 | 33 | 25 | −2 [−6, 2] |
| pvocab b3 | 845 | 77 | 53 | 39 | collapses |
| deployed chain (q8 fusion 0.1 + Qwen-0.5B re-rank, app45) | 81 (7.26 %, reproduced) | 50 | 24 | 20 | −31 [−48, −16] |

Pvocab leaves the percent clips as they were. It was never tried on the chain, never with common words or phrases, and never
as a re-rank bonus. Step 1 tests that variant.

## Step 1: full-vocabulary + phrase bias, n-best re-rank bonus

- **List per fold.** Every non-stopword word of home_train (203–224 words) plus every 2–4-gram that occurs in at least 2
  home_train clips (54–68 phrases). "percent" and the number words are in every fold's list.
- **App list.** None was added. Neither the app nor the v1.8 package defines one; it has only the user-entered "My vocabulary",
  which is empty by default.
- **Scoring.** fused = base + s·(list words + list phrases). The strength is picked leave-one-fold-out (LOFO) from
  {0.5, 1, 2}.

| arm | errors / 1116 | unseen-word err (/231) | "percent" clips exact (/8) | Δ vs no bias [95 % CI] | clips fixed / broken | do-no-harm F1 (v34_val 43; s3+s3ov val 65)* |
|---|---|---|---|---|---|---|
| F1, no bias | 112 (10.04 %) | 32 | 0 | — | — | 23; 24 |
| **F1 + bias, LOFO (s = 2 on all folds)** | **105 (9.41 %)** | 31 | 0 | **−7 [−17, 0]** | 6 / 2 | 23; 24 |
| F1 + bias s 0.5 / 1 | 111 / 111 | 32 / 32 | 0 | −1 [−5, 2] / −1 [−6, 3] | | 25; 24 / 25; 24 |
| chain, no bias | 81 (7.26 %) | 24 | 0 | — | — | n/a |
| **chain + bias, LOFO (s = 0.5 on all folds)** | **81 (7.26 %)** | 24 | 0 | **0 [−3, 3]** | 1 / 1 | n/a (needs App context) |
| chain + bias s 1 / 2 | 83 / 83 | 24 / 25 | 0 | +2 [−2, 6] / +2 [−4, 8] | | |

*Do-no-harm scores the scripted sets with my lower-case candidate keys. No bias gives 23 and 24 errors that way, where the
eval reports 20 and 20, because of number and contraction normalisation. Only compare within a column. At the LOFO strength
s = 2, F1 shows no harm on either set.

Seen/unseen breakdown for the LOFO arms: F1 seen 64 → 59 and unseen 32 → 31; chain seen 50 → 50 and unseen 24 → 24. Last-word
errors: F1 24 → 23, chain 20 → 20.

**Primary verdicts (pre-registered rule: CI upper bound must be < 0):**
- F1 alone: −7 errors, CI [−17, 0]. **Not shown to help**, though the trend favours it. This is about the same as or a
  little better than pvocab b1 (−3).
- Deployed chain: 0 [−3, 3]. **No effect.** The Qwen re-rank with app45 context already captures what a home-vocabulary
  bonus adds.

## Why the "percent" clips don't move

8 in-home clips have "percent" in the reference: "twenty / ninety / eighty-five / seventy-five / one hundred percent". In
every one, the top-1 hypothesis simply **drops the word at the end**. The digits are correct, so this is the end-of-clip
deletion seen before, not an out-of-vocabulary problem.

In 7 of the 8 clips the correct "… PERCENT" candidate is in the 20-best (one clip never has it). It sits at rank 2–13, but
**6–9 nats** below the top on beam score alone (7–9 nats in the chain). In the chain, 2 of the 7 rank 11–13, outside the
top-10 re-rank pool. A list bonus of at most about 2–6 per candidate, as tested, cannot close that gap. A bonus large enough
to do it (≈ 8) is the regime where pvocab b3 collapsed.

So "percent" does not look like a decode-bias problem. The visual model gives the final word almost no evidence, likely
because the clip ends or the mouth closes early. This fits the earlier finding that last-word errors are not formatting
artefacts. A text-to-mouth generator would have to beat F1 at 105/1116 and the chain at 81/1116. The bias gives no extra bar
on the chain.

## Files

- `README.md`: pre-registration (step 0 summary + frozen design).
- `step0_rescore.py`: rescoring of pvocab, chain and plain predictions, with the list leakage check (read-only).
- `pbias.py`: the step 1 bias re-rank, LOFO, bootstrap and do-no-harm.
- On torch: `/scratch/pa2753/synvid/pbias/step0.json` and `step1.json`, which include the per-percent-clip hypotheses.
