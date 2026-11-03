# G2 gate: ProtoDisent (IndexTTS-1.5 + pathology prefix + GRL) vs Higgs (`asr_aug/gen/gate_g1.py`, GEN=g2)

Kept clips, pools A+B. zs_wer capped at 1 (zero-shot whisper-large-v3). Real = that patient's real train role.

| voice | keep rate Higgs | keep rate G2 | G2 >= 0.5 x Higgs |
|---|---|---|---|
| old | 0.770 | 0.798 | True |
| s1 | 0.383 | 0.702 | True |
| s2 | 0.876 | 0.969 | True |

| patient | gen | n kept | zs_wer (real) | sec/word (real) | spk_sim | pause frac (real) |
|---|---|---|---|---|---|---|
| s1 | Higgs | 765 | 0.916 (0.909) | 0.916 (0.865) | 0.890 | 0.016 (0.007) |
| s1 | G2 | 1402 | 0.524 (0.909) | 0.875 (0.865) | 0.680 | 0.138 (0.007) |
|  | closer than Higgs (wer, rate, sim) | [False, True, False] | keep ok True | **FAIL** | | |
| s2 | Higgs | 1751 | 0.398 (0.269) | 0.891 (0.883) | 0.953 | 0.011 (0.006) |
| s2 | G2 | 1937 | 0.199 (0.269) | 0.809 (0.883) | 0.822 | 0.058 (0.006) |
|  | closer than Higgs (wer, rate, sim) | [True, False, False] | keep ok True | **FAIL** | | |

**Gate: s1 FAIL, s2 FAIL. G2 does not go to Step B (decided by s1).**
