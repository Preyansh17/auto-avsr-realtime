# V1 text/audio-exposure control — results (2026-11-10)

Pre-registration: `asr_aug/synvid/README.md` amendment A15. Same split, recipe, seeds and test T as V0 (A14).

- A2 = A0 (583 real videos) + 441 audio-only TTS clips (Higgs, 3 healthy voices) of 160 of the 169 S sentences
  (66/507 clips failed the Whisper ≤ 0.10 check; 9 sentences had no passing clip). Trained through the
  audio branch of the shared USR 2.0 encoder (`usr2_ft_v1.py`, audio rows → ctc_a / out_layer_a).
- Audio sanity: base checkpoint reads Higgs speech at 0.58% greedy WER (A15 limit 25%) → runnable.

| W-token errors, WiSE 0.75, sum seeds 1-3 (360 tokens) | errors | vs A0 [95% CI] |
|---|---|---|
| A0 real video without S | 25 | |
| A1 + real S video | 18 | -7 [-15, -1] |
| A2 + S as TTS audio only | 24 | -1 [-5, +2] |

A2 - A1 = +6 [+1, +14]. By seed: A2 7/9/8. All-token WER on T: A0 8.66, A1 7.72, A2 8.47.

Recovery r = (A0 - A2) / 7 = 0.14 < 0.5 → **A15 outcome: generator justified** (V2 = A13 step 1 only on Preyansh's go).
Reading: hearing the new words (shared encoder) gives almost none of the gain; the gain in V0 needs the
mouth movements of the word. Cost ≈ 1.5 GPU-h (cap 6).
