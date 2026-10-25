# Experiment RS: reference study (fold 0, 46 sentences, 36 with PER and ArtP in every row; seeds [1, 2, 3])

Real PER 0.720, ArtP 0.179, TitaNet ceiling 0.680. `bias` = mean per-sentence (synthetic - real), `MAE` = mean absolute per-sentence PER error, `d MAE` = MAE minus the ft_stitch5 row of the same model (paired bootstrap 95% CI over sentences; negative = better than ft_stitch5).

## References

| reference | seconds | clips | distinct words | origin |
|---|---|---|---|---|
| ft_stitch5 | 27.7 | 5 | 22 | new, tts_train only |
| stitch5 | 25.8 | 5 | 22 | existing zero-shot reference, all sources in fold-0 training set |
| lowpitch5 | 25.6 | 5 | 15 | existing zero-shot reference, all sources in fold-0 training set |
| rs_rand1 | 7.6 | 1 | 7 | 1 random clip >= 6 s |
| rs_rand3 | 14.3 | 3 | 12 | 3 random clips |
| rs_rand5a | 27.4 | 5 | 24 | 5 random clips, draw a (disjoint sentences from ft_stitch5) |
| rs_rand5b | 24.5 | 5 | 19 | 5 random clips, draw b (disjoint from a) |
| rs_vocab5 | 38.7 | 5 | 36 | greedy: 5 clips maximising distinct words |

## Finetuned B_r64 (fold 0, epoch 8)

| reference | PER | PER bias | PER MAE | d MAE vs ft_stitch5 [95% CI] | ArtP | ArtP bias | TitaNet | within +-20% | ramble |
|---|---|---|---|---|---|---|---|---|---|
| ft_stitch5 | 0.693 +- 0.054 | -0.080 | 0.185 | - | 0.198 | +0.038 | 0.664 | 65% | 0.0% |
| stitch5 | 0.725 +- 0.024 | -0.032 | 0.220 | +0.035 [-0.045, +0.152] | 0.162 | -0.005 | 0.661 | 78% | 0.0% |
| lowpitch5 | 0.689 +- 0.007 | -0.083 | 0.180 | -0.005 [-0.039, +0.030] | 0.159 | -0.001 | 0.668 | 78% | 0.0% |
| rs_rand1 | 0.700 +- 0.032 | -0.068 | 0.187 | +0.002 [-0.034, +0.040] | 0.175 | +0.012 | 0.670 | 73% | 0.7% |
| rs_rand3 | 0.692 +- 0.015 | -0.086 | 0.168 | -0.016 [-0.054, +0.022] | 0.180 | +0.023 | 0.671 | 76% | 0.0% |
| rs_rand5a | 0.657 +- 0.009 | -0.111 | 0.170 | -0.015 [-0.046, +0.016] | 0.173 | +0.004 | 0.660 | 71% | 0.7% |
| rs_rand5b | 0.691 +- 0.020 | -0.074 | 0.168 | -0.016 [-0.049, +0.016] | 0.181 | +0.015 | 0.680 | 75% | 0.0% |
| rs_vocab5 | 0.728 +- 0.008 | -0.042 | 0.172 | -0.013 [-0.058, +0.043] | 0.162 | -0.003 | 0.672 | 75% | 0.0% |

## Zero-shot (pretrained, same references)

| reference | PER | PER bias | PER MAE | d MAE vs ft_stitch5 [95% CI] | ArtP | ArtP bias | TitaNet | within +-20% | ramble |
|---|---|---|---|---|---|---|---|---|---|
| ft_stitch5 | 0.744 +- 0.051 | -0.012 | 0.208 | - | 0.193 | +0.030 | 0.603 | 41% | 8.7% |
| stitch5 | 0.814 +- 0.061 | +0.042 | 0.186 | -0.022 [-0.081, +0.038] | 0.176 | +0.012 | 0.642 | 38% | 13.0% |
| lowpitch5 | 0.602 +- 0.023 | -0.141 | 0.259 | +0.051 [-0.022, +0.126] | 0.328 | +0.166 | 0.620 | 41% | 13.0% |
| rs_rand1 | 0.700 +- 0.032 | -0.054 | 0.245 | +0.037 [-0.018, +0.092] | 0.253 | +0.088 | 0.479 | 35% | 5.1% |
| rs_rand3 | 0.732 +- 0.060 | -0.028 | 0.238 | +0.031 [-0.051, +0.117] | 0.250 | +0.088 | 0.597 | 49% | 8.7% |
| rs_rand5a | 0.585 +- 0.041 | -0.177 | 0.349 | +0.141 [+0.038, +0.275] | 0.306 | +0.144 | 0.586 | 54% | 3.6% |
| rs_rand5b | 0.753 +- 0.058 | +0.033 | 0.242 | +0.034 [-0.031, +0.096] | 0.270 | +0.106 | 0.611 | 45% | 8.7% |
| rs_vocab5 | 0.772 +- 0.037 | +0.015 | 0.229 | +0.022 [-0.057, +0.101] | 0.217 | +0.059 | 0.599 | 54% | 10.9% |

## Sensitivity to the reference (spread across the 8 references)

| model | PER bias range | PER bias sd | MAE range | TitaNet range | within +-20% range | ramble range | mean seed sd of PER |
|---|---|---|---|---|---|---|---|
| finetuned B | 0.080 | 0.025 | 0.052 | 0.020 | 12 pts | 0.7 pts | 0.021 |
| zero-shot | 0.220 | 0.080 | 0.163 | 0.162 | 20 pts | 9.4 pts | 0.045 |

Reference-draw noise: finetuned B, rand5a vs rand5b differ by 0.037 in PER bias (seed sd of PER 0.021).

