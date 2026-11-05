# G3 gate: VoxCPM2 + LoRA vs Higgs (`asr_aug/gen/gate_g1.py`, GEN=g3)

Kept clips, pools A+B. zs_wer capped at 1 (zero-shot whisper-large-v3). Real = that patient's real train role.

| voice | keep rate Higgs | keep rate G3 | G3 >= 0.5 x Higgs |
|---|---|---|---|
| old | 0.770 | 0.157 | False |
| s1 | 0.383 | 0.146 | False |
| s2 | 0.876 | 0.116 | False |

| patient | gen | n kept | zs_wer (real) | sec/word (real) | spk_sim | pause frac (real) |
|---|---|---|---|---|---|---|
| s1 | Higgs | 765 | 0.916 (0.909) | 0.916 (0.865) | 0.890 | 0.016 (0.007) |
| s1 | G3 | 291 | 0.024 (0.909) | 0.497 (0.865) | 0.629 | 0.163 (0.007) |
|  | closer than Higgs (wer, rate, sim) | [False, False, False] | keep ok False | **FAIL** | | |
| s2 | Higgs | 1751 | 0.398 (0.269) | 0.891 (0.883) | 0.953 | 0.011 (0.006) |
| s2 | G3 | 231 | 0.017 (0.269) | 0.508 (0.883) | 0.623 | 0.163 (0.006) |
|  | closer than Higgs (wer, rate, sim) | [False, False, False] | keep ok False | **FAIL** | | |

**Gate: s1 FAIL, s2 FAIL. G3 does not go to Step B (decided by s1).**
