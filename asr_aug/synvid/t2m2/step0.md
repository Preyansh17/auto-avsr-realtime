# A13 targeted text-to-mouth, step 0 (2026-11-10)

All outputs on torch under `/scratch/pa2753/synvid/t2m2/` (pool/, crops/, align/, text/, step0_stats.json). No
transcripts or Tianyu-derived files leave torch; this file holds counts only. Compute: CPU jobs on `cs` (srun 22518866,
22518919, 22519041, 22519472 +2 short; devcrop sbatch 22518967) and one GPU job (align 22518968, H100, 1 min 05 s ≈ 0.02 GPU-h).

## 1. Training pool (`pool/pool.jsonl`, `pool/pool_report.json`)
Source split: `semg3_rcv_20261109_v1` roles (the deployed-chain in-home CV split).

| source | listed | dropped | kept (readable) | held-out clips | generator-train clips |
|---|---|---|---|---|---|
| s3_train | 240 | 0 | 240 | 8 | 232 |
| s3ov_train | 262 | 0 | 262 | 9 | 253 |
| v3.4 train (Tianyu's split) | 298 | 0 | 298 | 9 | 289 |
| real conversation (dc_real_all) | 117 | 5 | 112 | 4 | 108 |
| **total** | 917 | 5 | **912** | 30 | **882** |

- Eval overlap: no pool clip is a clip of s3_val, s3ov_val, s3_test, s3_oov, s3ov_test, v3.4 val/test (and _abl
  copies), rainbow or any in-home role (dc_home_all/dev/val/test); identity checked on devcrop_id, sample_id,
  phrase_id, video basename and (domain, take_id). Bare take_ids collide across sessions (3 s3ov_train takes share
  ids with s3_oov takes recorded on 10-15; different clips), so identity is session-qualified.
- The 5 dropped real clips have text equal (normalised) to an in-home clip (one-to-three-word phrases, incl. a name);
  no other pool clip matches any eval sentence.
- Raw video: s3 session raw mp4 (3 are `_redo` takes), s3ov session raw mp4 (8 takes have no raw file: their 25 fps
  `_normalized_25p` copy is used), v3.4 phone .mov from the SEMG3_HPC package (via the aligned_v1 index), real
  `videos/NNN.mov` (rl_NNN = labels.csv index; every label text checked against the role transcript).
- Note: Tianyu's `derived/devcrop_v1` (s3/s3ov/real crops) and its manifest are not readable to pa2753; v34 crops are.
- 30 held-out reconstruction sentences: `random.Random(20261113)`, quotas s3 8 / s3ov 9 / v34 9 / real 4, >= 3 words,
  listed in `pool/heldout30.txt`; every pool clip with one of these texts is excluded from generator training (30 clips).
  10 texts occur on more than one pool clip.

## 2. Crops (`crops/npz`, deliverable devcrop.py, MediaPipe 0.10.21, v18crop args)
- Success 912/912 (100 %); 69,684 frames (s3 19,452, s3ov 22,199, v34 21,744, real 6,289). Median length s3 3.20 s,
  s3ov 3.32 s, v34 2.84 s, real 1.88 s. Missing-face frames: 0 except 3 real clips (any missing).
- Spot check, 10 v34 clips (seed 20261113) against Tianyu's `devcrop_v34_v1`: same T on all 10, offset 0, mean abs grey
  diff < 0.001 (rounds to 0.000), 96-100 % of frames bit-identical. No other pre-existing devcrops of pool clips are
  readable.

## 3. Timings (`align/video_align.jsonl`, `align/timings.json`)
Video-CTC forced alignment of the known transcript (USR 2.0 Huge VSR CTC head, align_words.py method, devcrop reader
`t2m2/align_npz.py`): 912/912 aligned, 0 errors. Word boundaries = first-unit frame, gap midpoints, last-unit frame;
phones spread inside words by the A11 average-duration table (CMUdict first pronunciation).

| | s3 | s3ov | v34 | real | all |
|---|---|---|---|---|---|
| s / word, median (p10-p90) | 0.24 (0.20-0.29) | 0.26 (0.18-0.37) | 0.24 (0.18-0.32) | 0.24 (0.13-0.37) | 0.24 (0.18-0.33) |
| lead-in, CTC (median) | 0.64 | 0.64 | 0.68 | 0.40 | 0.64 (0.44-0.80) |
| tail, CTC (median) | 1.20 | 1.12 | 0.60 | 0.24 | 0.96 (0.32-1.36) |
| lead-in, aperture onset (median) | 0.08 | 0.20 | 0.08 | 0.00 | 0.12 (0.00-0.32) |
| tail, aperture offset (median) | 0.80 | 0.84 | 0.36 | 0.00 | 0.64 (0.08-0.96) |

- Word duration median 0.22 s (p10 0.12, p90 0.42; 5,604 words).
- Per-phone durations (median, s): vowels AA .105 AE .104 AH .073 AO .113 AW .13 AY .14 EH .086 ER .104 EY .116 IH .075
  IY .096 OW .12 OY .146 UH .077 UW .12; consonants mostly .05-.095 (full p10/p50/p90 table in step0_stats.json).
  These are the prior table rescaled by each word's measured length, so they carry word-level timing only, not
  measured phone timing.
- Caveat: CTC lead-in is ~0.5 s later than the aperture onset (peaky CTC fires after the lips start moving), so CTC
  word starts run late; the duration model should take lead-in/tail from the aperture spans and only the
  inter-word structure from CTC. Real conversation clips are cut at release (tail ~0) - lead-in/tail must be modelled
  per source.
- G2P: CMUdict misses 4 pool word types (4 tokens). g2p_en 2.1.0 is installed in `/scratch/pa2753/envs/idx2` (and
  `envs/pdis`) and runs offline through `G2p.predict(word)`; `G2p()(sentence)` fails because NLTK
  `averaged_perceptron_tagger_eng` is not on torch (only the old `averaged_perceptron_tagger`). For single OOV words
  predict() is enough; the homograph path would need that one NLTK resource (download, ask first).

## 4. Vocabulary
- Pool: 1,086 word types (882 generator-train clips: 1,055). By source: s3 50, s3ov 612, v34 615, real 262.
  F1 (deployed model) training text = s3 + s3ov + real117: 747 types.
- In-home 193 clips (1,116 tokens, deployed chain reproduced at 81 errors):

| tokens | n | in pool video | already in F1 training video | new via v3.4 only | not in pool video |
|---|---|---|---|---|---|
| all in-home | 1,116 | 972 | 933 | 39 | 144 |
| unseen-word (vs fold home_train) | 231 | 121 | 97 | 24 | 110 (103 types) |
| chain unseen-word errors | 24 | 8 | 6 | 2 | 16 |

- **Key number:** real pool video covers 121/231 unseen-word tokens (52 %), but 97 of those are words F1 was already
  trained on in real video; v3.4 adds only 24 tokens (2 of the 24 chain errors). 110 tokens (103 types) and 16 of
  the 24 chain unseen-word errors are on words with no real video anywhere in the pool - that is the generator's only
  target. Descriptive: chain error rate on unseen tokens is 6/97 when F1 trained on the word, 16/110 when the word
  has no pool video.

## 5. Text pool (`text/dd_pool.jsonl`, `text/dd_greedy500.jsonl`)
- Source: DailyDialog (CC BY-NC-SA 4.0), already on torch in the 2026-10-31 text package
  (`patients/texts/dialog_corpora_v20261031`); nothing downloaded. 17 other corpora sit there too (ED, Taskmaster, ...).
- Filters: dialogs with any test/val material dropped (478 of 13,118); turns flagged same-text-as-test/val dropped
  (12,767); sentences of 3-10 normalised words (93,210); exact leak (lower-cased raw text) 9 and normalised leak (eval
  normaliser, also with spaces removed) 92 dropped against 5,890 / 7,568 eval texts (all rcv eval roles incl. the 193
  in-home references, LATEST_combined eval roles, and all 8,918 package val/test rows with their keys); 1 held-out
  sentence dropped; 2,272 with a CMUdict-OOV word; 28,681 with no word missing from generator-train video. A final
  re-check asserts no written sentence matches any eval text.
- **Pool: 48,838 distinct sentences covering 10,475 word types missing from pool video.** A greedy 500-sentence subset
  (max new words) covers 2,078 of them.
- Selection does not use in-home words. Descriptive only: of the 111 in-home unseen tokens with no generator-train
  video, 98 occur in the text pool and 55 in the greedy 500.

## Files (repo asr_aug/synvid/t2m2/)
`pool.py` (1), `devcrop.sbatch` (2), `align_npz.py` + `align.sbatch` (3), `stats.py` (2-5), `g2p_check.py` (3).
