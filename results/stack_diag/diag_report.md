# Step 0: Higgs synthetic vs real patient clips (`asr_aug/stack/diag_synth.py`)

Means (median in brackets). zs_wer = zero-shot whisper-large-v3 WER, higher = sounds more impaired.

## s1

| set | n | zs_wer | sec_per_word | pause_frac | spk_sim |
|---|---|---|---|---|---|
| real train | 240 | 1.059 [1.000] | 0.865 [0.845] | 0.007 [0.006] | 0.921 [0.931] |
| real val | 30 | 0.986 [1.000] | 0.873 [0.822] | 0.006 [0.005] | 0.901 [0.921] |
| synth A kept | 314 | 0.933 [1.000] | 0.870 [0.845] | 0.016 [0.013] | 0.890 [0.900] |
| synth A rejected | 686 | 1.023 [1.000] | 0.872 [0.847] | 0.017 [0.012] | 0.894 [0.907] |
| synth B kept | 451 | 0.987 [1.000] | 0.949 [0.928] | 0.016 [0.013] | 0.890 [0.898] |
| synth B rejected | 547 | 1.079 [1.000] | 0.986 [0.960] | 0.018 [0.013] | 0.892 [0.901] |

## s2

| set | n | zs_wer | sec_per_word | pause_frac | spk_sim |
|---|---|---|---|---|---|
| real train | 240 | 0.302 [0.143] | 0.883 [0.861] | 0.006 [0.005] | 0.951 [0.961] |
| real val | 30 | 0.378 [0.226] | 0.845 [0.858] | 0.006 [0.005] | 0.956 [0.965] |
| synth A kept | 815 | 0.462 [0.375] | 0.834 [0.787] | 0.011 [0.010] | 0.954 [0.962] |
| synth A rejected | 185 | 0.670 [0.714] | 0.840 [0.792] | 0.017 [0.009] | 0.939 [0.961] |
| synth B kept | 936 | 0.400 [0.333] | 0.941 [0.912] | 0.011 [0.010] | 0.952 [0.961] |
| synth B rejected | 62 | 0.812 [0.817] | 1.017 [1.040] | 0.013 [0.011] | 0.928 [0.950] |

## Decision rule (fixed before reading): synthetic easier if kept pool-A zs_wer <= real-train - 10 pp

- s1: real-train zs_wer 1.059 vs kept pool-A 0.933 (gap +12.6 pp) -> EASIER (rule met)
- s2: real-train zs_wer 0.302 vs kept pool-A 0.462 (gap -16.0 pp) -> not easier by the rule
