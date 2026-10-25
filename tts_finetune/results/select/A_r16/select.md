# Checkpoint selection: A_r16

> **tts_val has only 20 sentences (22 real clips). n=20. PER differences below ~0.05 are noise;** gap SE is the standard error of the per-sentence (synth - real) PER difference. This is a screen for collapse / rambling / voice loss, not a fine ranking.

Rule (pre-registered): lowest |PER_synth - PER_real| subject to ramble rate <= zero-shot stitch10 ramble rate (same sentences) and TitaNet voice match >= 0.60; ties -> earlier epoch.
Selection reference: `ft_stitch5` (built only from tts_train clips). Seeds: [1]. Split hash 9341ad8cb2cb.
Ramble rate = share of sentences whose synthetic clip is > 2x the real length; within +-20% = ratio in [0.8, 1.25]. Zero-shot stitch10 ramble rate on these sentences: 40% (8/20).

| epoch | ref | PER | |PER-real| | gap SE | ArtP | TitaNet | WavLM | dur ratio med | within +-20% | ramble (>2x) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real |  | 0.657 | 0 |  | 0.182 | 0.687 (ceiling) | 0.906 (ceiling) | 1.00 | 100% | 0% | 20 sent / 22 clips |
| zero-shot stitch10 | R1b_stitch10_s1 | 0.822 | 0.165 |  | 0.161 | 0.653 | 0.894 | 1.41 | 45% | 40% (8/20) | ramble yardstick |
| 0 | ft_stitch5 | 0.825 | 0.168 | 0.082 | 0.181 | 0.634 | 0.906 | 1.21 | 40% | 0% (0/20) | pretrained, same ref (control) |
| 1 | ft_stitch5 | 0.932 | 0.280 | 0.073 | 0.074 | 0.436 | 0.665 | 0.09 | 16% | 0% (0/19) | eliminated: only 19.0/20.0 sentences generated; TitaNet 0.436 < 0.60 |
| 2 | ft_stitch5 | 0.785 | 0.128 | 0.078 | 0.147 | 0.698 | 0.905 | 1.12 | 45% | 0% (0/20) | eligible |
| 3 | ft_stitch5 | 0.681 | 0.024 | 0.061 | 0.170 | 0.671 | 0.904 | 1.05 | 55% | 0% (0/20) | eligible (within noise of chosen) |
| 5 | ft_stitch5 | 0.689 | 0.032 | 0.052 | 0.146 | 0.684 | 0.927 | 1.01 | 60% | 0% (0/20) | eligible (within noise of chosen) |
| 8 | ft_stitch5 | 0.742 | 0.085 | 0.071 | 0.138 | 0.688 | 0.917 | 1.03 | 60% | 0% (0/20) | eligible |
| 12 | ft_stitch5 | 0.675 | 0.018 | 0.061 | 0.136 | 0.700 | 0.892 | 0.95 | 50% | 0% (0/20) | **CHOSEN** |
| 15 | ft_stitch5 | 0.769 | 0.112 | 0.067 | 0.102 | 0.670 | 0.874 | 1.04 | 50% | 0% (0/20) | eligible |

**Chosen: epoch 12** (|PER-real| = 0.018, ramble 0%, TitaNet 0.700) -> `/scratch/pa2753/higgs_ft/runs/A_r16/epoch_12`
Eligible epochs within 0.05 PER of the chosen one (not distinguishable at n=20): [3, 5]. The rule still picks the lowest gap; look at their audio before committing.
