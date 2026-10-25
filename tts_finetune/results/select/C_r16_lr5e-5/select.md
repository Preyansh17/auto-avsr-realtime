# Checkpoint selection: C_r16_lr5e-5

> **tts_val has only 20 sentences (22 real clips). n=20. PER differences below ~0.05 are noise;** gap SE is the standard error of the per-sentence (synth - real) PER difference. This is a screen for collapse / rambling / voice loss, not a fine ranking.

Rule (pre-registered): lowest |PER_synth - PER_real| subject to ramble rate <= zero-shot stitch10 ramble rate (same sentences) and TitaNet voice match >= 0.60; ties -> earlier epoch.
Selection reference: `ft_stitch5` (built only from tts_train clips). Seeds: [1]. Split hash 9341ad8cb2cb.
Ramble rate = share of sentences whose synthetic clip is > 2x the real length; within +-20% = ratio in [0.8, 1.25]. Zero-shot stitch10 ramble rate on these sentences: 40% (8/20).

| epoch | ref | PER | |PER-real| | gap SE | ArtP | TitaNet | WavLM | dur ratio med | within +-20% | ramble (>2x) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real |  | 0.657 | 0 |  | 0.182 | 0.687 (ceiling) | 0.906 (ceiling) | 1.00 | 100% | 0% | 20 sent / 22 clips |
| zero-shot stitch10 | R1b_stitch10_s1 | 0.822 | 0.165 |  | 0.161 | 0.653 | 0.894 | 1.41 | 45% | 40% (8/20) | ramble yardstick |
| 0 | ft_stitch5 | 0.825 | 0.168 | 0.082 | 0.181 | 0.634 | 0.906 | 1.21 | 40% | 0% (0/20) | pretrained, same ref (control) |
| 1 | ft_stitch5 | 0.964 | 0.307 | 0.063 | 0.055 | 0.342 | 0.537 | 0.05 | 0% | 0% (0/20) | eliminated: TitaNet 0.342 < 0.60 |
| 2 | ft_stitch5 | 0.749 | 0.092 | 0.058 | 0.133 | 0.638 | 0.880 | 1.04 | 40% | 0% (0/20) | eligible |
| 3 | ft_stitch5 | 0.781 | 0.124 | 0.051 | 0.128 | 0.680 | 0.901 | 0.95 | 55% | 0% (0/20) | eligible |
| 5 | ft_stitch5 | 0.772 | 0.115 | 0.065 | 0.139 | 0.691 | 0.910 | 0.99 | 45% | 0% (0/20) | eligible |
| 8 | ft_stitch5 | 0.773 | 0.116 | 0.062 | 0.125 | 0.667 | 0.931 | 1.02 | 65% | 0% (0/20) | eligible |
| 12 | ft_stitch5 | 0.636 | 0.021 | 0.052 | 0.165 | 0.662 | 0.911 | 1.08 | 70% | 0% (0/20) | **CHOSEN** |
| 15 | ft_stitch5 | 0.737 | 0.080 | 0.052 | 0.165 | 0.678 | 0.913 | 0.98 | 55% | 0% (0/20) | eligible |

**Chosen: epoch 12** (|PER-real| = 0.021, ramble 0%, TitaNet 0.662) -> `/scratch/pa2753/higgs_ft/runs/C_r16_lr5e-5/epoch_12`
