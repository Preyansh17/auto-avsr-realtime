# pbias — decode-only phrase bias for in-home app words (pre-registration)

Written 2026-11-10, before any bias-scored output was looked at. Cluster outputs: `/scratch/pa2753/synvid/pbias/` (torch).
Purpose: the bar a text-to-mouth generator would have to beat later, using decode-time biasing only (no training).

## Step 0 finding (why step 1 runs)

Tianyu's `pvocab` (runs/s3v34/home_eval_v1/bp/pvocab_rcv, script `s3v34_pvocab_bp.sbatch`, scorer `s3d_bias.py`) is an
in-beam per-token bonus while a hypothesis spells a listed word (taken back if the word is not finished), on **F1 plain beam
only** (not the deployed chain), bonus {1, 2, 3} per sub-word token. Its per-fold list is the **rare** words (not in a general
top-5000 list) of that fold's home_train transcripts: 21–27 words, all of them in the fold's home_train (no leakage). Common
words such as "percent" and most number words are excluded by the rarity filter, so the "percent" clips are not targeted.
Result: 112 → 109 errors / 1116 (b1, CI [−7, 0]); b3 collapses (845). It never ran on the chain. So step 1 tests a variant
he has not tried: full home_train vocabulary + phrases, as an n-best re-ranking bonus, on both F1 alone and the deployed chain.

## Frozen design

**Data.** `semg3_rcv_20261109_v1` random 5-fold in-home CV; 193 clips (`rcv{k}_home_val`, k = 0..4), 1116 reference words.
References = the eval's own `normalized_ground_truth` (numbers spelled out). No decoding of any kind is done (0 GPU-hours);
all arms re-rank n-best lists already decoded by Tianyu. test53, test57, OOV50 and in-home test48 are not touched.

**Models** (v1.8 F1, `semg3_usr2_F1_a075_20261027.pth`, beam 10, CTC 0.1):
- `F1` — F1 alone: n-best of `bp/pvocab_rcv/F1/b0/f{k}` (plain decode, = `decgrid_F1/c0.1_b10`), first 20 distinct candidates,
  score = beam score.
- `chain` — deployed v1.8 chain: n-best of `bp/chain/F1_app45_q8/fus_lm0.1_lb0` (in-beam Qwen word fusion 0.1, app45
  context, q8 LM) re-ranked as delivered: first 10 distinct candidates, score = vsr + 0.5 · Qwen2.5-0.5B lm(c | app45) + 0 ·
  n_words (`lm/F1_chainq8_v3/Qwen2.5-0.5B.json`). Reproduces the reported 81/1116 = 7.26 %.

**Bias list per fold k** (built from `rcv{k}_home_train` normalised transcripts only; never from that fold's home_val):
- words: every distinct word of home_train that is not in a fixed English function-word stoplist (`STOP` in `pbias.py`);
- phrases: every word 2–4-gram that occurs in ≥ 2 home_train clips and contains at least one non-stoplist word.
- App list: the Convey app / delivery package defines no fixed vocabulary list (v1.8 config has only the user-entered
  "My vocabulary", empty by default), so **no app list is added**. "percent" and number words enter only if they are in
  that fold's home_train text.

**Method.** n-best re-ranking (fits the delivered eval code; mirrors the v1.8 "My vocabulary" rescoring bonus):
`fused(c) = base(c) + s · (n_words_in_list(c) + n_phrases_in_list(c))`, counting every occurrence; ties keep the original order.

**Strength.** s ∈ {0.5, 1, 2}. Reported arm = leave-one-fold-out: for fold k, s is the value with the fewest pooled word errors
on the other four folds' home_val (ties → smaller s), then applied to fold k. Fixed-s results are reported too (descriptive).

**Primary** (one per model, both reported): pooled word errors on all 193 home_val clips, LOFO arm vs the same model without
bias; paired bootstrap over clips (10 000 resamples, seed 0), 95 % CI of the error difference. "Helps" only if the CI's upper
bound is < 0.

**Secondary** (descriptive): errors on reference words seen vs unseen in the fold's home_train; the clips whose reference
contains "percent" (count and how many become exact); last-word errors; n-best oracle (reference in candidate pool).

**Do-no-harm** (report only): F1 alone with the deployment list (same rules, built from all 193 in-home transcripts) at each
s, on `dc_v34_val` (43) and `dc_s3_val` + `dc_s3ov_val`, minus any clip that is also an in-home clip. The chain needs App
context, which scripted sets do not have, so it is not run there.

**Stop.** Write `results.md`, report, stop. No further tuning on these 193 clips.
