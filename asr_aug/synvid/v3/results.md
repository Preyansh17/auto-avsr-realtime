# V3 (A17) result — generated lip video of support sentences, 2026-11-11

Jobs 22701373/74/75 (A100, ~43 min each, ~2.1 GPU-h). Test = 100 T clips, 360 W-word tokens (3 seeds summed).
Score: `/scratch/pa2753/synvid/v3/score_a0.75.json` (`v3_score.py`).

| arm | train | W errors (a0.75, primary) | by seed | all-token T WER | W errors a1.0 (descriptive) |
|---|---|---|---|---|---|
| A0 | pool − T − S | 25 | 8/9/8 | 8.66 | 27 |
| A1 | + real S video | 18 | 5/6/7 | 7.72 | 17 |
| A2 | + S as TTS audio | 24 | 7/9/8 | 8.47 | 26 |
| A3 | + generated S video | 22 | 7/7/8 | 8.19 | 20 |

Primary (a0.75): A3 − A0 = −3, 95% CI [−7, 0]; recovery (A0−A3)/7 = 0.43; A3 < A2 yes (22 vs 24).
A3 − A1 = +4 [−2, +12]; A3 − A2 = −2 [−6, +2].
**Decision: FAIL** (CI upper = 0, not < 0; recovery 0.43 < 0.5).

Descriptive only (not pre-registered primary, cannot rescue the decision): at a1.0 A3 − A0 = −7 [−14, −1], recovery 0.70.

Reading: generated video moves W errors in the right direction (about 40-70% of the real-video gain depending on
WiSE alpha), unlike TTS audio (r = 0.14), but the primary test misses by the narrowest margin. Even a full real-video
gain is worth ~4-5 of 81 in-home chain errors.
