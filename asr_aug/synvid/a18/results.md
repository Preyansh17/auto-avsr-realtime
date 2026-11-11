# A18 result — dose + replication of V3, 2026-11-11

Jobs: gen 22715363, manifest 22715364, A0+A1 22715365/72/95, A3x1 22715366/76/96, A3x5 22715368/80/97 (all COMPLETED,
1,000 updates each; some preempted and requeued, resumed from checkpoint). Test = 100 T clips, W tokens, seeds 4-6 summed.
Score: `/scratch/pa2753/synvid/a18/score_a0.75.json` (`a18_score.py`).

A3x5 manifest: 504 generated clips = 169 S + 320 DailyDialog (leak check dropped 30) + 15 S re-samples (gen seeds 2-5,
10 W words); every W word >= 5 clips (105 at exactly 5).

| arm | W errors a0.75 (primary) | by seed 4/5/6 | other-unseen errors (/378) | all-token T WER | W errors a1.0 |
|---|---|---|---|---|---|
| A0 | 23 | 6/9/8 | 67 | 8.24 | 22 |
| A1 real S video | 18 | 4/7/7 | 62 | 7.39 | 19 |
| A3x1 169 gen | 18 | 5/7/6 | 68 | 8.15 | 20 |
| A3x5 504 gen | 18 | 7/6/5 | 68 | 8.80 | 21 |

Primary A3x5 - A0 = -5 [-12, 0]. A3x1 - A0 = -5 [-12, 0]. A3x5 - A3x1 = 0 [-3, 3]. A3x5 - A1 = 0 [-4, 4].
A1 - A0 (real-video oracle) = -5 [-12, 0] — the oracle itself also misses CI < 0 on these seeds.
**Frozen decision: CLOSE** (primary CI upper = 0, and A3x5 no better than A3x1). a1.0 also "close" (A3x5 - A0 -1 [-9, 5]).

Reading:
- Dose: no effect. 3x more generated video per new word gives the same W errors as 1x (0 [-3, 3]) and slightly worse
  all-token WER (8.80 vs 8.15).
- Replication: A3x1 matches real video on seeds 4-6 (18 vs 18). Pooled seeds 1-6 (descriptive): A0 48, A1 36,
  A3x1 40 -> generated video recovers ~2/3 of the real-video gain.
- The test cannot separate effects of this size: even real video is only -5 [-12, 0] on 3 seeds. Real ceiling is
  ~4-5 of 81 in-home errors, so more seeds would buy little.
- Harm check: generated arms add ~1 other-unseen error vs A0 (68 vs 67); no harm.
