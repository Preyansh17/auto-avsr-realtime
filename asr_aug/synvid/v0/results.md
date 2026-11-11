# V0 real-video oracle — results (2026-11-10)

Pre-registration: `asr_aug/synvid/README.md` amendment A14. Split `/scratch/pa2753/synvid/v0/split/` (seed 20261114).
Runs `/scratch/pa2753/synvid/v0/runs/{A0,A1}/seed{1,2,3}`; scores `/scratch/pa2753/synvid/v0/score_a{0.75,1.0}.json`.

- A0 = 583 pool clips with no held-out word (W, 118 types). A1 = A0 + 169 real clips containing W words (S).
- Test T = 100 clips, 120 W tokens per seed. F1 recipe, 1,000 updates, model alone, beam 10, CTC 0.1.

| metric (sum over seeds 1-3) | A0 | A1 | A1 - A0 [95% CI] |
|---|---|---|---|
| W-token errors, WiSE 0.75 (primary) | 25/360 (6.9%) | 18/360 (5.0%) | -7 [-15, -1] |
| W-token errors, a1.0 | 27/360 | 17/360 | -10 [-19, -3] |
| other unseen tokens, a0.75 | 70/378 | 70/378 | 0 |
| all-token WER on T, a0.75 | 8.66% | 7.72% | |

By seed (a0.75): A0 8/9/8, A1 5/6/7.

**Verdict: PASS** (CI upper < 0). Real video of a new word lowers errors on that word. Gain is
word-specific (other unseen words unchanged). Size: ~28% of W errors removed (≈2.3 errors per
120 W tokens per seed). A1 also adds the sentences' text, so pixels vs text exposure is not yet
separated — that is V1.

Cost ≈ 2.4 GPU-h (cap 12).
