# semg3 targeted text-to-mouth plan (sections 1-8, as relayed 2026-11-10)

Source: cloud project files; copied verbatim from the coordinator brief. Pre-registered as amendment A13 in README.md.

## Bar
Deployed chain: 81/1116 word errors on the 193 in-home clips; 24 of them are unseen-word errors (of 231 unseen-word
tokens).

## Generator
DualLip-style: text -> phonemes -> semg3 duration model -> 96x96 grey devcrop mouth crops in a frozen autoencoder
latent, conditioned on one reference frame. Losses: L1 + perceptual; a frozen lip-reader CTC loss from a reader that
is NOT the judge (Auto-AVSR LRS3 VSR or USR 2.0 Base); smoothness.

## Data rules (sec 2)
Training pool: readable labelled semg3 clips in s3_train (240), s3ov_train (262), v3.4 train (298, Tianyu's split)
and real conversation (~116). No clip from any eval set (s3_val, s3ov_val, s3_test, s3_oov, s3ov_test, v3.4 val/test,
rainbow, any in-home clip). 30 held-out reconstruction sentences (fixed seed, spread across sources) excluded from
generator training. Generation text: public conversational English, 3-10 words, exact + normalised leak check
against every eval transcript including the 193 in-home ones. /scratch/th3482 read-only; outputs under
/scratch/pa2753/synvid/t2m2/; nothing from Tianyu's files leaves torch.

## Gate G4 (sec 4), frozen
Judge: USR 2.0 Huge zero-shot; v1.8 descriptive.
- (a) 30 held-out reconstruction sentences: readability (mean capped WER) <= 0.372.
- (b) 100 new pool sentences: readability <= 0.372, crop validity >= 0.98.
- (c) sharpness >= 0.7 of real; likeness inside the real spread.
Stop rule: (a) > 0.6 after the first full run -> stop.

## Training test (sec 5)
Only if G4 passes and Preyansh says go. F1 recipe, R vs R+S (synthetic <= 50% of real, <= 500 clips), 3 seeds, chain
decode. Primary: pooled in-home word errors on the 193 clips with the deployed chain, R+S vs R, seed-paired
bootstrap; "helps" = CI upper bound < 0.

## Budget
Step 0 CPU (plus <= 0.5 GPU-h video-CTC alignment); step 1 8-12 GPU-h; G4 < 1 GPU-h.

## Steps
0. Pool, crops, timings, vocabulary coverage, text pool (step0.md). Stop and report.
1. Train generator (only on go).
2. G4.
3. Training test (only on go).
