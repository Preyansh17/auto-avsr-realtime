# End-of-clip check — results (2026-11-10)

Pre-registration: README.md (written before any decoding). Model: delivered v1.8 (semg3_usr2_F1_a075_20261027.pth), model alone,
beam 10, CTC 0.1, nbest 5 / keep 20, delivered usr2_eval.py with one change (repeat the last frame PAD_FRAMES times; code/usr2_eval_pad.py).
No LM chain. Not decoded: test53, test57, OOV50, in-home test48 (as a set). Paired bootstrap over clips, 10,000 resamples, seed 0.
Machines: step 0 login node; step 1 CPU node job 22501873 (cs); step 2 GPU jobs 22501990 (4 min) + 22502511 (A100, 27 min) = ~0.52 GPU-h.

## Step 0 — scoring check
- In-home references: "percent" as a word 8 times (8 clips; 7 end in it), "%" 0 times, digit tokens 17 (14 clips; the normaliser spells them out).
- F1 predictions: "percent" 0, "%" 0, digit tokens 18.
- The delivered normaliser deletes "%" (punctuation), so "90%" -> "ninety"; but no "%" occurs on either side, so a consistent normaliser
  ("%" -> "percent") gives identical scores: last-wrong 24, first-wrong 10, 112/1116 (bootstrap difference 0 [0, 0]).
- The 7 "dropped percent" errors are real model errors, not formatting: all 7 in-home clips ending in "percent" lose it.
  F1's training targets contain "percent" in 1 clip (dc_s3ov_train); F1 never emits the word.
- Gap after removing the 7 percent clips: 17 last-wrong vs 10 first-wrong (of 186 clips). Rest of the gap is mixed final-word
  substitutions (quit/it, bigger/peer, cold/gold, score/sword, car/card, park/party, ...).

## Step 1 — tails (devcrop npz the model reads; phone-front-end aperture, lip_activity.py velocity/span rule, threshold 0.25 x pooled p95 on the 193 clips)
| | n | lead-in median (s) | tail median (s) | tail p10-p90 (s) |
|---|---|---|---|---|
| all | 193 | 0.36 | 0.28 | 0.08-0.52 |
| last word wrong | 24 | 0.26 | 0.26 | 0.09-0.63 |
| last word wrong, non-percent | 17 | 0.36 | 0.24 | 0.10-0.66 |
| last word right | 169 | 0.36 | 0.28 | 0.08-0.52 |

Tail last-wrong vs right: Mann-Whitney p = 0.76 (AUC 0.48); non-percent p = 0.98. Tail < 0.2 s: 8/24 wrong vs 47/169 right.
Deviation from brief: aperture is the phone front end's inner-lip/mouth-width signal stored in each crop, not a new FAN run
(the FAN s3_train threshold is on another scale).

Raw vs processed: crops span the whole raw .mov (crop minus video duration median -0.008 s, within one frame). Recording stops a fixed
0.30 s after button release (post-roll); crop end is 0.29 s after release (median). Clips are cut at recording end (release + 0.3 s),
not trimmed later. No raw frames exist past the clip end -> arm (d) not possible.

## Step 2 — re-decode
| arm | in-home last-word-wrong | first-word-wrong | errors/words | s3ov_val errors (fu_v1 crops*) | last-wrong diff vs (a) [95% CI] | errors diff vs (a) [95% CI] |
|---|---|---|---|---|---|---|
| (a) no pad | 24 | 10 | 112/1116 (10.04%) | 24/210 | — | — |
| (b) +0.2 s last frame | 24 | 10 | 115/1116 (10.30%) | 25/210 | 0 [0, 0] | +3 [0, +7] |
| (c) +0.4 s last frame | 23 | 10 | 115/1116 (10.30%) | 27/210 | -1 [-3, 0] | +3 [-1, +8] |
| (d) +0.3 s from raw | n/a — no raw frames past clip end | | | | | |

Reproduction: in-home arm (a) matches F1's predictions exactly (0 clips with different text, 112 = 112).
Padding changed 9 of 193 in-home clips; none of the 7 percent clips.

*s3ov_val: the delivered devcrop_v1 crops are not readable for pa2753. Decoded on three readable front-end variants; none reproduces
D0 (20/210): fi_v1 27, fu_v1 24, fv_v1 26 in arm (a). Per the stop rule, the do-no-harm column is **unverified**; descriptively, padding
never helped on any variant (fi 27/27/27, fu 24/25/27, fv 26/26/26 for a/b/c).

## Conclusion
The last-word excess is not a scoring or clip-cut artefact. 7 of 24 last-word errors are the untrained word "percent"; the clips already
keep ~0.3 s after release and tails do not differ by error status; frame-repeat padding does not reduce last-word errors (24 -> 24 -> 23)
and adds ~3 errors. Fixing "percent" needs training text (spoken-form targets / home data, cf. rh4sp) or an LM/vocab bias, not end padding.
