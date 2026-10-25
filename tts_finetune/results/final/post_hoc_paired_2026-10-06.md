srun: job 19308749 queued and waiting for resources
srun: job 19308749 has been allocated resources
Test sentences: 40; near-duplicate of a tts_train sentence (word edit distance <= 1): 15; far: 25.

## PER: per-sentence (synth - real), n sentences, bootstrap 95% CI over sentences

| condition | n | bias (mean d) | |bias| | MAE (mean abs d) | MAE 95% CI | near (n) bias / MAE | far (n) bias / MAE |
|---|---|---|---|---|---|---|---|
| A_r16 e12 | 40 | +0.014 | 0.014 | 0.205 | [0.164, 0.249] | (15) +0.063 / 0.252 | (25) -0.016 / 0.177 |
| B_r64 e8 | 40 | -0.005 | 0.005 | 0.235 | [0.170, 0.314] | (15) +0.031 / 0.393 | (25) -0.026 / 0.140 |
| C_r16_lr5e-5 e12 | 40 | +0.009 | 0.009 | 0.217 | [0.148, 0.318] | (15) +0.107 / 0.326 | (25) -0.050 / 0.152 |
| stitch5 (3 seeds) | 40 | +0.062 | 0.062 | 0.215 | [0.166, 0.269] | (15) +0.097 / 0.212 | (25) +0.041 / 0.217 |
| stitch10 | 40 | +0.084 | 0.084 | 0.269 | [0.209, 0.335] | (15) +0.043 / 0.285 | (25) +0.108 / 0.260 |
| lowpitch5 | 40 | -0.264 | 0.264 | 0.350 | [0.266, 0.435] | (15) -0.314 / 0.338 | (25) -0.235 / 0.357 |
| median1 | 40 | -0.245 | 0.245 | 0.455 | [0.360, 0.555] | (15) -0.316 / 0.445 | (25) -0.202 / 0.461 |

Paired vs zero-shot stitch5 on the same sentences (PER), FT minus stitch5; negative = FT closer to real:

| FT run | d|bias| (|mean d_FT| - |mean d_s5|) [95% CI] | d MAE (mean|d_FT| - mean|d_s5|) [95% CI] | frac sentences closer to real |
|---|---|---|---|
| A_r16 e12 | -0.048 [-0.103, +0.046] | -0.010 [-0.068, +0.046] | 50% |
| B_r64 e8 | -0.057 [-0.116, +0.086] | +0.019 [-0.067, +0.116] | 42% |
| C_r16_lr5e-5 e12 | -0.053 [-0.116, +0.083] | +0.002 [-0.085, +0.112] | 52% |

## ARTP: per-sentence (synth - real), n sentences, bootstrap 95% CI over sentences

| condition | n | bias (mean d) | |bias| | MAE (mean abs d) | MAE 95% CI | near (n) bias / MAE | far (n) bias / MAE |
|---|---|---|---|---|---|---|---|
| A_r16 e12 | 40 | -0.028 | 0.028 | 0.087 | [0.068, 0.108] | (15) -0.027 / 0.090 | (25) -0.029 / 0.086 |
| B_r64 e8 | 40 | -0.019 | 0.019 | 0.079 | [0.059, 0.100] | (15) -0.022 / 0.078 | (25) -0.018 / 0.080 |
| C_r16_lr5e-5 e12 | 40 | -0.026 | 0.026 | 0.074 | [0.056, 0.097] | (15) -0.038 / 0.077 | (25) -0.019 / 0.073 |
| stitch5 (3 seeds) | 40 | -0.008 | 0.008 | 0.113 | [0.093, 0.135] | (15) -0.033 / 0.109 | (25) +0.008 / 0.115 |
| stitch10 | 40 | -0.008 | 0.008 | 0.145 | [0.113, 0.180] | (15) -0.012 / 0.173 | (25) -0.005 / 0.128 |
| lowpitch5 | 40 | +0.148 | 0.148 | 0.183 | [0.139, 0.228] | (15) +0.167 / 0.200 | (25) +0.136 / 0.173 |
| median1 | 40 | +0.203 | 0.203 | 0.247 | [0.195, 0.300] | (15) +0.218 / 0.248 | (25) +0.194 / 0.247 |

Paired vs zero-shot stitch5 on the same sentences (ARTP), FT minus stitch5; negative = FT closer to real:

| FT run | d|bias| (|mean d_FT| - |mean d_s5|) [95% CI] | d MAE (mean|d_FT| - mean|d_s5|) [95% CI] | frac sentences closer to real |
|---|---|---|---|
| A_r16 e12 | +0.020 [-0.025, +0.040] | -0.026 [-0.048, -0.004] | 57% |
| B_r64 e8 | +0.012 [-0.028, +0.031] | -0.034 [-0.055, -0.013] | 68% |
| C_r16_lr5e-5 e12 | +0.019 [-0.024, +0.037] | -0.039 [-0.059, -0.018] | 68% |
