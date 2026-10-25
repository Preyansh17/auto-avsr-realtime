# Higgs v2 TTS augmentation: similarity check

**Question:** does zero-shot cloning of the patient's voice sound like the patient (tone, pace, patterns, rhythm)? Scope: TTS only, no finetuning and no ASR training yet.

## What we did

- Built the missing measurements (the package had none).
  - Voice match with two independent speaker models.
  - 25 pace/pitch/rhythm measures.
  - Word errors.
  - PathBench articulatory precision (run in a second session).
  - Every metric is judged against a real-vs-real baseline, meaning two halves of the real clips.
- Covered all 7 generation runs, about 30,000 clips.
  - Patient: one reference, stitch5, stitch10.
  - TORGO 8 speakers: zero-shot, stitch5, stitch10 capped.
- Ran a blind listening test (1 listener, 24 clips).

## Results

| Area | Finding |
|---|---|
| **Voice** | Stitched references fix it. Patient TitaNet match goes 0.51 -> **0.70 / 0.71** (real-vs-real ceiling 0.69). F04 and M04 stay well below ceiling under every strategy. |
| **Pitch, average pace** | Inherited from the reference clip. Synthetic pitch lands within a few Hz of the reference, and references sit 14-25 Hz above the real speakers' averages. |
| **Per-sentence pace** | Uncontrolled. Only 41% of sentences fall within +/-20% of the real length. |
| **Severe speakers, zero-shot** | Too clean. Word error, real minus synthetic: M01 +0.47, M04 +0.43, M02 +0.38. PathBench agrees: severe real 0.33 vs synthetic 0.52 (mild matches: 0.62 vs 0.63). Speech is also faster (severe: 2.41 vs 1.81 syllables/s). This replicates the F5-TTS paper. |
| **Stitched runs** | Gaps shrink, but the transcriber invents words (M04 word error 0.62 -> 1.01). Stitch10 scores lower (0.17) than the patient's own PathBench score (0.23), possibly reflecting errors more than dysarthria. |
| **Patient baseline, PathBench** | Precision indistinguishable from real (0.226 vs 0.232, p = 0.53). Speech rate and pitch variability match too. |
| **Listening test** | Real clips accepted as same speaker 11/12, synthetic 2/12. Voice, pace and rhythm averaged 3.0, 2.75 and 2.75 for synthetic vs 4.33 for real. Weakest setup only (single reference). |
| **Prolongation, repetition** | No signal, inside real-vs-real noise. |

## Caveats

- One listener and 12 synthetic clips.
- TORGO synthetic sentences are twice as long as real ones (13 vs 7 words), which confounds word errors.
- F01 has only 20 real clips.
- 9 real M05 clips are empty audio and were excluded.
- Severity labels differ between runs (M05 is "mild" in one manifest, "Moderate" in the config).
- PathBench's DArtP and PER need a 35 GB language model that was not downloaded.

## Next

**No GPU needed**

1. Listening test on stitched clips plus F04, M04 and M01, and a second listener.
2. PathBench on the stitched TORGO runs.
3. Agree the "similar enough" rule before seeing more results.

**GPU, needs approval**

4. Patient sentences with stitched references, for a paired comparison.
5. References chosen to match each speaker's median pitch and pace.
6. Seed-noise reruns.
7. Sampling and prompt tests (`ras_win_len`, temperature, system prompt).

**Decision after these:** tune the settings or finetune the TTS.

*Files (cluster): `/scratch/pa2753/tts_eval/out_v2/` and `/scratch/pa2753/pb_eval/out/`.*
