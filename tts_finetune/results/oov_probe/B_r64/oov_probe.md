# OOV generalization probe: FT_B_r64_best_ft_stitch5_oov_sentences_s1

**OOV PROBE PASS: n=50/50 new sentences, seed 1 | ramble FT 0/50 vs zero-shot 7/50 (ok, limit +4%) | short+silent FT 0/50 (ok, limit 4%) | PER FT 0.638 vs zero-shot 0.703, paired delta -0.065 [-0.151, +0.020] (ok, limit +0.15)**

Sentences: 50 from `/scratch/pa2753/higgs_ft/data/oov_sentences.json`; 0 of them equal a patient or ASR sentence (checked against 348 known patient sentences). PER mode: computed. Mean fraction of tokens unseen in the 73-word training vocabulary: 0.79.

## Definitions

- Expected duration = 1.82 + 0.526 x words (OLS on 283 real tts_train clips; residual sd 1.07 s; real sec/word ratio-of-sums 0.85).
- RAMBLE: duration > 2 x expected. SHORT: < 0.5 x expected. SILENT: rms < 0.005. `ramble (chars)` is the same rule with expected duration fit on characters (sensitivity to longer words).
- PER: PathBench PEREvaluator against the clip's OWN target text (there is no ground truth for new sentences).
- Verdict limits: ramble <= zero-shot + 4%; short+silent <= 4%; mean paired PER(FT) - PER(zero-shot) <= +0.15.

## Arms (same text, same reference, 1 seed)

| metric | FT_B_r64_best_ft_stitch5_oov_sentences_s1 [ft] | FT_zs_ft_stitch5_oov_sentences_s1 [zs] |
|---|---:|---:|
| clips | 50 | 50 |
| ramble (>2x expected) | 0 (0%; 95% CI 0%-7%) | 7 (14%; 95% CI 7%-26%) |
| ramble (chars) | 0 | 2 |
| short (<0.5x) | 0 | 0 |
| silent | 0 | 0 |
| within +-20% of expected | 50% | 30% |
| duration / expected: median [p10, p90] | 1.19 [0.92, 1.45] | 1.42 [0.95, 2.14] |
| sec per word: mean (median) [p10, p90] | 0.98 (0.95) [0.69, 1.27] | 1.29 (1.14) [0.78, 1.90] |
| longest clip (s) | 10.8 | 34.9 |
| PER mean (median, sd) | 0.638 (0.694, 0.18) n=50 | 0.703 (0.753, 0.26) n=50 |

Paired PER delta FT_B_r64_best_ft_stitch5_oov_sentences_s1 - FT_zs_ft_stitch5_oov_sentences_s1: -0.065 (bootstrap 95% CI [-0.151, +0.020], 50 sentences).

## By word count

| group | n | FT_B_r64_best_ft_stitch5_oov_sentences_s1: ramble / PER | FT_zs_ft_stitch5_oov_sentences_s1: ramble / PER |
|---|---:|---:|---:|
| 4-5 words | 20 | 0/20 / 0.66 | 3/20 / 0.69 |
| 6-8 words | 20 | 0/20 / 0.59 | 3/20 / 0.74 |
| 9-12 words | 10 | 0/10 / 0.70 | 1/10 / 0.66 |

## By share of unseen words (median split)

| group | n | FT_B_r64_best_ft_stitch5_oov_sentences_s1: ramble / PER | FT_zs_ft_stitch5_oov_sentences_s1: ramble / PER |
|---|---:|---:|---:|
| lower-OOV half | 25 | 0/25 / 0.60 | 4/25 / 0.68 |
| higher-OOV half | 25 | 0/25 / 0.67 | 3/25 / 0.73 |

## Caveats

- One seed, 50 sentences: one clip is 2 points of ramble rate and the PER delta CI is wide. This is a screening probe for memorisation/collapse, not a ranking; the headline comparison is the 3-seed tts_test evaluation.
- PER here is an ASR-phoneme mismatch against the target text, so it also rises with harder (rarer, longer) words; read it as FT relative to zero-shot on identical text, not against the patient's real PER (0.665 on in-domain text).
- Ramble is duration-only. A fluent but wrong utterance (a memorised patient sentence said instead of the target) shows up as high PER, not as ramble: inspect the highest-PER FT clips by ear.
- Per-clip table: `/scratch/pa2753/higgs_ft/oov_probe/B_r64_best_ft_stitch5_oov_sentences_s1/oov_clips.csv`; PathBench items (for pb_score_items.py cross-check): `/scratch/pa2753/higgs_ft/oov_probe/B_r64_best_ft_stitch5_oov_sentences_s1/oov_pb_items.csv`.
