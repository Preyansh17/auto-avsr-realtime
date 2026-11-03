# G1D gate: IndexTTS-2 + LoRA + duration control vs Higgs (`asr_aug/gen/gate_g1.py`, GEN=g1d)

Kept clips, pools A+B. zs_wer capped at 1 (zero-shot whisper-large-v3). Real = that patient's real train role.

| voice | keep rate Higgs | keep rate G1D | G1D >= 0.5 x Higgs |
|---|---|---|---|
| old | 0.770 | 0.565 | True |
| s1 | 0.383 | 0.237 | True |
| s2 | 0.876 | 0.484 | True |

| patient | gen | n kept | zs_wer (real) | sec/word (real) | spk_sim | pause frac (real) |
|---|---|---|---|---|---|---|
| s1 | Higgs | 765 | 0.916 (0.909) | 0.916 (0.865) | 0.890 | 0.016 (0.007) |
| s1 | G1D | 473 | 0.824 (0.909) | 0.814 (0.865) | 0.764 | 0.007 (0.007) |
|  | closer than Higgs (wer, rate, sim) | [False, True, False] | keep ok True | **FAIL** | | |
| s2 | Higgs | 1751 | 0.398 (0.269) | 0.891 (0.883) | 0.953 | 0.011 (0.006) |
| s2 | G1D | 967 | 0.681 (0.269) | 0.852 (0.883) | 0.957 | 0.004 (0.006) |
|  | closer than Higgs (wer, rate, sim) | [False, False, True] | keep ok True | **FAIL** | | |

**Gate: s1 FAIL, s2 FAIL. G1D does not go to Step B (decided by s1).**
