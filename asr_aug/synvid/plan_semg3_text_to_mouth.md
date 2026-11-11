# semg3 text-to-mouth plan (local copy)

Source: project file plans/semg3_text_to_mouth_plan.md. This machine cannot read the project Library, so this copy is
built from the coordinator's step-1 summary of 2026-11-09. Replace it with the full plan text when available.

## Step 1: splice test (CPU only; outputs under /scratch/pa2753/synvid/t2m/)
1. Source units: s3_train only, minus 24 held-out clips (fixed seed, listed in a file); existing 96x96 mouth crops at
   25 fps; word timings from the existing alignments; phonemes from CMUdict spread inside each word by average phone
   duration.
2. Unit inventory: whole-word segments, plus diphone segments (phone midpoint to next phone midpoint) for unseen words.
3. Per target sentence: unit selection with Viterbi; join cost = pixel/landmark distance at the boundary frames plus
   mean-brightness difference; target cost prefers whole words. 2-3 frame cross-fade. Short real closed-mouth segment
   from s3_train at start and end.
4. Two sets: (a) the 24 held-out sentences (text only, their clips excluded); (b) the 100 D2 pool sentences. Report
   the in-vocabulary word rate.
5. Gate G3: USR 2.0 Huge zero-shot readability <= 0.372; sharpness >= 0.7 of real; crop validity; patient likeness
   (lip-opening range, open/close speed, sec/word from mouth landmarks vs the real s3_train spread); v1.8 WER as a
   descriptive second judge only. Compare with D2 G-LSft on the same 100 sentences.
6. Stop rule: set (a) readability > 0.6 -> stop and report.

Rules: no val/test/oov/s3ov_test/rainbow clips; nothing from Tianyu's deliverable leaves torch; scripts as files;
CPU on the cs partition (account torch_pr_39_general); Huge and v1.8 scoring in a short GPU job (< 0.5 GPU-h).
Write /scratch/pa2753/synvid/t2m/step1.md, report, stop. The frozen G3 details are README amendment A11.
