# G1 gate: IndexTTS-2 + LoRA vs Higgs (`asr_aug/gen/gate_g1.py`)

Kept clips, pools A+B. zs_wer capped at 1 (zero-shot whisper-large-v3). Real = that patient's real train role.

| voice | keep rate Higgs | keep rate G1 | G1 >= 0.5 x Higgs |
|---|---|---|---|
| old | 0.770 | 0.502 | True |
| s1 | 0.383 | 0.227 | True |
| s2 | 0.876 | 0.355 | False |

| patient | gen | n kept | zs_wer (real) | sec/word (real) | spk_sim | pause frac (real) |
|---|---|---|---|---|---|---|
| s1 | Higgs | 765 | 0.916 (0.909) | 0.916 (0.865) | 0.890 | 0.016 (0.007) |
| s1 | G1 | 454 | 0.825 (0.909) | 1.353 (0.865) | 0.913 | 0.035 (0.007) |
|  | closer than Higgs (wer, rate, sim) | [False, False, True] | keep ok True | **FAIL** | | |
| s2 | Higgs | 1751 | 0.398 (0.269) | 0.891 (0.883) | 0.953 | 0.011 (0.006) |
| s2 | G1 | 709 | 0.468 (0.269) | 1.364 (0.883) | 0.972 | 0.013 (0.006) |
|  | closer than Higgs (wer, rate, sim) | [False, False, True] | keep ok False | **FAIL** | | |

**Gate: s1 FAIL, s2 FAIL. G1 does not go to Step B (decided by s1).**
