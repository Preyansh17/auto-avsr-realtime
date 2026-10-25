# Experiment R: reference sensitivity

229 held-out sentences from the K folds (189 with PER and ArtP in every row), seeds [1, 2, 3]. Real PER 0.679, ArtP 0.205, TitaNet ceiling 0.692. Criteria R1/R2 are pre-registered (README, Experiment R).

## Reference: ft_stitch5

| row | PER | bias | ArtP | TitaNet | within +-20% | ramble |
|---|---|---|---|---|---|---|
| Finetuned A | 0.710 +- 0.012 | +0.032 | 0.165 | 0.681 | 75% | 0.1% |
| Finetuned B | 0.695 +- 0.020 | +0.018 | 0.177 | 0.670 | 73% | 0.0% |
| Finetuned C | 0.737 +- 0.016 | +0.061 | 0.158 | 0.687 | 72% | 0.1% |
| Zero-shot, same reference | 0.741 +- 0.020 | +0.066 | 0.198 | 0.611 | 46% | 6.8% |

## Reference: stitch5

| row | PER | bias | ArtP | TitaNet | within +-20% | ramble |
|---|---|---|---|---|---|---|
| Finetuned A | 0.722 +- 0.009 | +0.045 | 0.158 | 0.686 | 74% | 0.3% |
| Finetuned B | 0.706 +- 0.018 | +0.033 | 0.168 | 0.679 | 76% | 0.0% |
| Finetuned C | 0.726 +- 0.026 | +0.045 | 0.157 | 0.686 | 75% | 0.4% |
| Zero-shot, same reference | 0.797 +- 0.033 | +0.128 | 0.184 | 0.650 | 42% | 13.7% |

## Reference: lowpitch5

| row | PER | bias | ArtP | TitaNet | within +-20% | ramble |
|---|---|---|---|---|---|---|
| Finetuned A | 0.697 +- 0.016 | +0.024 | 0.163 | 0.679 | 77% | 0.0% |
| Finetuned B | 0.677 +- 0.014 | +0.002 | 0.178 | 0.681 | 76% | 0.0% |
| Finetuned C | 0.731 +- 0.007 | +0.055 | 0.158 | 0.689 | 77% | 0.0% |
| Zero-shot, same reference | 0.581 +- 0.024 | -0.086 | 0.326 | 0.626 | 45% | 11.9% |

## Reference: median1

| row | PER | bias | ArtP | TitaNet | within +-20% | ramble |
|---|---|---|---|---|---|---|
| Finetuned A | 0.733 +- 0.012 | +0.061 | 0.161 | 0.683 | 73% | 0.3% |
| Finetuned B | 0.713 +- 0.011 | +0.036 | 0.160 | 0.678 | 73% | 0.4% |
| Finetuned C | 0.747 +- 0.012 | +0.072 | 0.149 | 0.690 | 70% | 0.3% |
| Zero-shot, same reference | 0.498 +- 0.007 | -0.186 | 0.387 | 0.525 | 45% | 5.7% |

## R1: quality holds for each reference (bias <= 0.07, TitaNet >= 0.64, within +-20% >= 55%, ramble < same-reference zero-shot)

- Finetuned A: **PASS**  ft_stitch5 PASS; stitch5 PASS; lowpitch5 PASS; median1 PASS
- Finetuned B: **PASS**  ft_stitch5 PASS; stitch5 PASS; lowpitch5 PASS; median1 PASS
- Finetuned C: **FAIL**  ft_stitch5 PASS; stitch5 PASS; lowpitch5 PASS; median1 FAIL (PER)

## R2: finetuning reduces the spread of mean PER across references (range FT - range Z, 95% CI must be < 0)

| run | range FT | range Z | difference [95% CI] | verdict |
|---|---|---|---|---|
| Finetuned A | 0.037 | 0.314 | -0.277 [-0.330, -0.220] | **PASS** |
| Finetuned B | 0.034 | 0.314 | -0.280 [-0.334, -0.216] | **PASS** |
| Finetuned C | 0.027 | 0.314 | -0.287 [-0.340, -0.224] | **PASS** |

## Secondary: per reference, finetuned vs same-reference zero-shot (paired over sentences, 95% CI)

| run | reference | |bias| FT - Z | per-sentence MAE FT - Z |
|---|---|---|---|
| Finetuned A | ft_stitch5 | -0.034 [-0.068, -0.000] | -0.039 [-0.064, -0.015] |
| Finetuned A | stitch5 | -0.083 [-0.123, -0.045] | -0.057 [-0.091, -0.026] |
| Finetuned A | lowpitch5 | -0.062 [-0.122, +0.007] | -0.089 [-0.117, -0.062] |
| Finetuned A | median1 | -0.125 [-0.195, -0.053] | -0.123 [-0.163, -0.085] |
| Finetuned B | ft_stitch5 | -0.048 [-0.080, -0.009] | -0.049 [-0.073, -0.025] |
| Finetuned B | stitch5 | -0.095 [-0.136, -0.052] | -0.049 [-0.086, -0.012] |
| Finetuned B | lowpitch5 | -0.084 [-0.115, -0.013] | -0.072 [-0.099, -0.046] |
| Finetuned B | median1 | -0.150 [-0.219, -0.075] | -0.129 [-0.167, -0.090] |
| Finetuned C | ft_stitch5 | -0.005 [-0.038, +0.027] | -0.051 [-0.076, -0.028] |
| Finetuned C | stitch5 | -0.083 [-0.123, -0.044] | -0.074 [-0.108, -0.044] |
| Finetuned C | lowpitch5 | -0.031 [-0.103, +0.040] | -0.067 [-0.098, -0.035] |
| Finetuned C | median1 | -0.114 [-0.190, -0.036] | -0.114 [-0.154, -0.075] |

## Spread across references (max - min of the four reference means)

| row | PER | ArtP | TitaNet | within +-20% |
|---|---|---|---|---|
| Finetuned A | 0.036 | 0.007 | 0.007 | 4 pp |
| Finetuned B | 0.036 | 0.017 | 0.011 | 3 pp |
| Finetuned C | 0.020 | 0.010 | 0.004 | 7 pp |
| Zero-shot, same reference | 0.299 | 0.203 | 0.124 | 4 pp |
