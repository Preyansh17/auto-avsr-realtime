# Checkpoint selection: B_r64

> **tts_val has only 20 sentences (22 real clips). n=20. PER differences below ~0.05 are noise;** gap SE is the standard error of the per-sentence (synth - real) PER difference. This is a screen for collapse / rambling / voice loss, not a fine ranking.

Rule (pre-registered): lowest |PER_synth - PER_real| subject to ramble rate <= zero-shot stitch10 ramble rate (same sentences) and TitaNet voice match >= 0.60; ties -> earlier epoch.
Selection reference: `ft_stitch5` (built only from tts_train clips). Seeds: [1]. Split hash 9341ad8cb2cb.
Ramble rate = share of sentences whose synthetic clip is > 2x the real length; within +-20% = ratio in [0.8, 1.25]. Zero-shot stitch10 ramble rate on these sentences: 40% (8/20).

| epoch | ref | PER | |PER-real| | gap SE | ArtP | TitaNet | WavLM | dur ratio med | within +-20% | ramble (>2x) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real |  | 0.657 | 0 |  | 0.182 | 0.687 (ceiling) | 0.906 (ceiling) | 1.00 | 100% | 0% | 20 sent / 22 clips |
| zero-shot stitch10 | R1b_stitch10_s1 | 0.822 | 0.165 |  | 0.161 | 0.653 | 0.894 | 1.41 | 45% | 40% (8/20) | ramble yardstick |
| 0 | ft_stitch5 | 0.825 | 0.168 | 0.082 | 0.181 | 0.634 | 0.906 | 1.21 | 40% | 0% (0/20) | pretrained, same ref (control) |
| 1 | ft_stitch5 | 0.806 | 0.148 | 0.047 | 0.128 | 0.662 | 0.899 | 1.10 | 50% | 0% (0/20) | eligible |
| 2 | ft_stitch5 | 0.741 | 0.084 | 0.075 | 0.128 | 0.659 | 0.920 | 0.97 | 50% | 0% (0/20) | eligible |
| 3 | ft_stitch5 | 0.732 | 0.074 | 0.054 | 0.131 | 0.646 | 0.918 | 1.03 | 60% | 0% (0/20) | eligible |
| 5 | ft_stitch5 | 0.687 | 0.029 | 0.068 | 0.168 | 0.674 | 0.927 | 1.13 | 60% | 0% (0/20) | eligible (within noise of chosen) |
| 8 | ft_stitch5 | 0.665 | 0.008 | 0.057 | 0.165 | 0.701 | 0.907 | 1.01 | 65% | 0% (0/20) | **CHOSEN** |
| 12 | ft_stitch5 | 0.720 | 0.063 | 0.042 | 0.134 | 0.697 | 0.916 | 0.99 | 75% | 0% (0/20) | eligible |
| 15 | ft_stitch5 | 0.631 | 0.027 | 0.062 | 0.172 | 0.702 | 0.906 | 1.03 | 65% | 0% (0/20) | eligible (within noise of chosen) |

**Chosen: epoch 8** (|PER-real| = 0.008, ramble 0%, TitaNet 0.701) -> `/scratch/pa2753/higgs_ft/runs/B_r64/epoch_08`
Eligible epochs within 0.05 PER of the chosen one (not distinguishable at n=20): [15, 5]. The rule still picks the lowest gap; look at their audio before committing.
