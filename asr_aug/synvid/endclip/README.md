# End-of-clip check (semg3 in-home clips) — pre-registration

Written 2026-11-10, before any decoding. Outputs live only under /scratch/pa2753/synvid/endclip/ on torch.

## Question
Notes in /scratch/pa2753/synvid/tianyu_runs_semg3.md report last-word errors ~2.4x first-word
errors on the 193 labelled in-home clips (F1: 24 vs 10 clips), many a dropped final "percent".
Is this (i) a scoring/normalisation artefact, (ii) clips cut too tight after the last lip movement,
or neither?

## Data
- Primary set: the 193 labelled in-home clips, package semg3-talk-labeled-20261107.
- Do-no-harm set: s3ov_val (35 clips).
- NOT decoded: test53, test57, OOV50, in-home test48.

## Model
v1.8 model from the deliverable, model alone (LM chain too only if easy), same crops and decoding as its eval.

## Arms
(a) no padding — must reproduce F1 predictions within 1-2 word errors, else stop and report.
(b) repeat last frame 0.2 s. (c) repeat last frame 0.4 s.
(d) only if raw videos have frames past the clip end: extend clip 0.3 s from raw.

## Endpoints
- Primary: number of in-home clips whose last reference word is wrong (alignment-based).
- Secondary: total word errors (S+D+I) on in-home; same two counts on s3ov_val.
- Also reported: first-word-wrong clips.
- Inference: paired bootstrap over clips (10,000 resamples, seed 0), 95% CI of the difference vs arm (a).

## Step 0 / Step 1 (CPU, before decoding)
- Step 0: count "percent", "%", digit forms in references and F1 predictions; check normaliser; if
  formatting explains errors, rescore with one consistent normaliser and report the remaining gap.
- Step 1: lead-in and tail after last lip movement (lip_activity.py / FAN inner-lip on model crops);
  compare tail length for clips with vs without last-word error; check raw recording length vs clip.

## Budget
<= ~0.5 GPU-hours, short jobs.
