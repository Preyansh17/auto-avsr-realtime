# TORGO extension of the Higgs synthetic-speech augmentation (step 6)

Does synthetic speech from per-speaker finetuned Higgs v2 adapters improve ASR on TORGO's 8 dysarthric speakers?
Decisions (user, 2026-10-15): one ASR model pooled over all 8 speakers; train on single words + sentences, report
them separately; Whisper large-v3 and Nemotron. Cluster mirror: `/scratch/pa2753/torgo_aug/` (code, data, higgs, root,
runs, logs, jobs.txt). Patient-study lessons applied: TTS never sees ASR val/test; judge is not a narrow-vocabulary model;
val-free final-epoch secondary; at most 8 concurrent Whisper runs.

## Pipeline

1. `build_splits.py`: 2822 head-mic clips (dropped 143 instruction/picture prompts, 3 without prompt, 30 < 0.2 s).
   Split by normalised prompt text across all speakers, 60/20/20 per type (seed 20261015): train 1693 / val 590 /
   test 539 clips; 0 texts shared between splits. Every test single word is therefore unseen in training.
2. `prep_tts.py`: per speaker, Higgs data from that speaker's ASR-train clips only (tts_val = 10% of its train texts,
   loss logging only); ft_stitch5 reference from 5 ASR-train clips. Asserted: no TTS clip or reference source is val/test.
3. `tts_train.sbatch`: patient B recipe (r64, alpha 128, lr 1e-4, 15-epoch cosine), adapter epoch 8 fixed in advance.
4. `build_text.py`: per speaker 160 words + 60 sentences. Words from TORGO-train + patient pool vocabularies, sentences
   from patient pool B. No text equals a val/test text, no sentence is within 1 word edit of a val/test sentence, no
   text contains a word found only in val/test (293 forbidden words).
5. `make_gen_plan.py` -> generation (`tts_finetune/final_gen.sbatch` stripes), `filter_synth.py collect`,
   `judge.sbatch` (zero-shot large-v3), `filter_synth.py score`. Thresholds: 95th pct judge CER of the same speaker's
   real train clips of the same type.
6. `make_labels.py`, `make_mix.py`, `whisper_torgo.sbatch` (large-v3, patient best hyperparameters, 15 epochs).

## Pre-registered (written 2026-10-15, before any TORGO ASR result)

- Arms: R (real train) and M0.5 (real + per speaker round(0.5 x real count) judge-kept synthetic). Seeds 1-7 each.
- Primary: test WER on all 539 test clips, M0.5 - R, val-selected best/. Seed-level two-sided Welch t-test on the 7 vs 7
  per-seed corpus WERs, and a sentence-level paired bootstrap (seed-averaged errors, 10000 draws) 95% interval.
  "Helps" only if both p < 0.05 and the bootstrap interval lies below 0.
- Secondary (same tests): final-epoch checkpoint; sentence-type split (sentence WER, single-word error rate);
  per speaker and per severity (mild F03 F04 M03 M05; severe F01 M01 M02 M04), reported without a claim.
- A "helps" claim needs the primary and the final-epoch secondary to agree.
- Reference point (no claim): zero-shot whisper-large-v3 on the same test set.
- Nemotron: same two arms after Whisper, 3+ seeds, current code (SaveBestWER fix), final-epoch and best reported.

## Progress

- 2026-10-15: 8 TTS adapters trained (6-12 min each, jobs 20002179-20002401); zero-shot judge on real train + test
  (20002433); R seeds 1-7 (20002453-460, ~1.5 h each).
- R test WER (best/): all 33.47 +- 2.63, sentences 23.54, words 60.89, mild 19.47, severe 54.71. Final epoch: 43.99
  (15 epochs overfit; val selection matters). Zero-shot large-v3: 52.55 (sentences 24.41, words 130.30: hallucinated
  continuations on isolated words; M04 222.97).
- 2026-10-16: generation 1760 clips (20018996/97).
- 2026-10-16: filter kept 1432/1758 (judge rejects 119, dur 68; silence rule 173, mostly words of M02/M03/M04 - may be too
  strict for single-word recordings, not checked against real word clips). M0.5 seeds 1-7 (20021229-235, ~1.9 h each);
  Nemotron R/M0.5 seeds 1-3, 39 epochs (chains 20021236-320).

## Result (2026-10-16): no detectable effect on either model

Test WER % (539 clips), mean +- sd over seeds; delta = M0.5 - R, Welch p over seeds, sentence bootstrap 95%.
Reports: `data/report_{whisper,nemotron}_{best,last}.md`.

| model / checkpoint | R | M0.5 | delta | Welch p | bootstrap 95% |
|---|---|---|---|---|---|
| **Whisper, best/ (primary)** | 33.47 +- 2.63 | 33.09 +- 1.62 | -0.38 | 0.75 | [-1.48, +0.73] |
| Whisper, final epoch | 43.99 +- 2.66 | 45.69 +- 5.84 | +1.70 | 0.50 | [-1.43, +6.56] |
| Nemotron, best.nemo | 29.44 +- 0.49 | 29.71 +- 1.01 | +0.27 | 0.71 | [-1.21, +2.08] |
| Nemotron, final epoch | 29.66 +- 0.89 | 28.52 +- 0.86 | -1.14 | 0.18 | [-2.72, +0.80] |

Pre-registered decision: M0.5 does not help (neither test passes, primary or final-epoch). Subsets (no claim): Whisper
severe -1.95 (p=0.30), mild +0.65; Whisper final-epoch words -4.55 [-6.64, -2.53] but sentences +3.96; Nemotron
final-epoch sentences -1.92 [-3.25, -0.70] (Welch p=0.055). These go in opposite directions across models and checkpoints;
with ~15 subset tests, isolated intervals excluding 0 are expected by chance. Nemotron beats Whisper on TORGO by ~4 pp
(29.4 vs 33.5); zero-shot large-v3 52.6.

## Post hoc (2026-10-16 to 19): recalibrated silence filter, Whisper M0.5sc

The fixed 0.3 speech-fraction floor rejected 31-37% of REAL single-word clips of M02/M03/M04, so it filtered speakers more
than bad generations. `filter_synth.py score --silence calibrated --tag _silcal` sets the floor per speaker x type at the 5th
percentile of real train clips (kept 1465 vs 1432; M0.5sc shares 578/846 synthetic clips with M0.5). 7 Whisper seeds
(jobs 20079278-284). Reports `data/report_whisper_{best,last}_sc.md`.

| checkpoint | R | M0.5sc | delta | Welch p | bootstrap 95% |
|---|---|---|---|---|---|
| best/ | 33.47 +- 2.63 | 32.10 +- 1.11 | -1.37 | 0.24 | [-2.37, -0.36] |
| final epoch | 43.99 +- 2.66 | 43.02 +- 1.22 | -0.97 | 0.41 | [-2.46, +0.46] |

Fails the pre-registered rule (needs both tests and agreement with the final epoch), and was chosen after seeing the main
result: suggestive at most. Sentences -1.65 [-2.79, -0.53], severe -2.41 [-4.45, -0.34] (subsets, no claim).
