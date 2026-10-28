# Tianyu's Notion "avsr" page: transcription (read 2026-10-26)

Source: https://app.notion.com/p/avsr-391e78899a908000acfad6fe6be269b9 (Tianyu He's workspace, guest access, browser pane).
Text blocks copied; numbers in images read by eye (± last digit). Splits are the shared tree (`LATEST_combined` roles):
Tianyu's Whisper-L3 zero-shot on semg1 new test30 is 85.2%, the same as ours, so the rows are identical.
Notation on the page: old = semg1 legacy+legal recordings, new¹ = semg1 Oct 1, new² = semg2.

## 1. Early work (July, patient split v1, seed 7, single runs)
- Offline Auto-AVSR (Conformer), constrained decoding, merged→merged: AV 21.3, A 25.8, V 37.8.
- Validation-selected baselines, open vocab, merged-trained merged test: Whisper-medium 10.31, large-v3 10.76,
  offline AV 14.35 (adaptive config 8.07), Auto-AVSR audio 16.59, RNN-T AV 32.7, RNN-T audio 35.0.
- Routine study (healthy-speaker recordings, front/down camera, simulated severity, whispered): AV helps only for whispered speech (unseen A ~0.41 → AV ~0.36).
- Streaming on H100: Nemotron CTC0.5 15.70% (TTFT p50 0.97 s); anchored Nemotron and Simul 11.21%. Mac/iPhone demo: 17.5-21.1% WER, TTFT about 0.5 s.

## 2. Higgs → Whisper-medium (replication of arXiv 2606.19823)
- TORGO: zero-shot 34.9, real 20.6, clone 31.2, hybrid 24.2 (stitch5 18.5), balanced hybrid 19.3. Paper: 31.6 / 24.4 / 26.0 / 25.1.
- "Patient" (set not named): zero-shot 63.1, real 19.6, hybrid 16.9, balanced 17.3-19.1.

## 3. Patient study (Oct, 3 seeds unless noted)
**Best so far.** semg1 test30 7.1 / OOV 12.4 and 4.1 / 10.0. Streaming: 1.04 s look-ahead 9.7 / 16.4, 80 ms 12.2 / 17.6.
semg2 test30 0.6 / OOV 2.7 (streaming 1.8 / 4.2 at 1.04 s; 2.8 / 6.7 at 80 ms).
**Recipes.** semg1: encoder-only, lr 1e-5, 30 epochs, plus old recordings (broader vocabulary). semg2: full FT.

**Model zoo (bar charts, semg1 new test30 / OOV49):**
- test30: best Auto-AVSR AV (new¹) 3.6, large-v3 (old+new¹) 4.7, Parakeet-TDT 6.5, large-v3 new¹ 6.5.
- OOV49: best Nemotron (old+new¹+new²) 18.4, Nemotron (old+new¹) 20.2, AV-HuBERT AV 21.5, large-v3 (old+new¹) 28.7.
- Off-the-shelf: everything over 70% on semg1. semg2 test30 0.6 for several finetuned models; OOV50 best Nemotron† 3.1.

**Ablations** (forest plots, 95% intervals, semg1 OOV49):
- Significantly worse: drop the video in AV-HuBERT (+13), encoder-only → LoRA (+12), encoder-only → full FT (+6), drop old recordings (+5), lr 2e-5.
- Null: add TORGO, AdamW → Adafactor, 60 epochs, lr 5e-6, synthetic pool 512 → 20k, add 512 general-text synthetic.
- On new-word recognition: weight averaging of 18 configs is the biggest gain (+15 points). Next: train only the first 16 encoder layers, long training with half Higgs synthetic, Canary or AV-HuBERT + 512 wide-text synthetic.

**Weight averaging and voting:**

| | single | weight avg | vote |
|---|---|---|---|
| semg1 test30 | 3.9 | 4.5 | 3.9 |
| semg1 OOV | 25.8 | 12.4 | 10.3 |
| semg2 test30 | 0.6 | 0.6 | 0.6 |
| semg2 OOV | 2.7 | 3.6 | 3.9 |

The page notes that errors are very consistent across Whisper variants.

**Data scaling.** semg1 flat by 60-180 sentences, both for new only and for old + N new. semg2 flat by 60-120.
Old data on semg1 OOV49: 0 → 25.6, half → 22.7, all → 20.8.

**Cross-patient.** Hard → easy transfers, easy → hard does not. semg2 trained alone on semg1 test30: 51.5 (L3), 72.8 (Nemotron).
TORGO alone does not help.

**Candidate retrieval (K candidates, hard vs random distractors).** semg1 OOV top-1 drops to about 45-65% at K ≥ 100.
AV holds up better than audio or video alone.

**Synthetic table (single runs, `*` = interval excludes 0), WER test30 / OOV, new-word recognition %:**

| semg1 | none | zero-shot Higgs | fine-tuned Higgs |
|---|---|---|---|
| Whisper full FT, semg1 new | 5.1 / 33.3 / 42.9 | 7.9 / 27.9 / 50.0 | 6.7 / 19.7* / 67.9* |
| Whisper enc-only, semg1 new | 5.1 / 25.8 / 52.4 | 7.7 / 23.0 / 56.0 | 8.3 / 18.8 / 73.8* |
| Whisper enc-only, both patients | 7.1 / 18.5 / 57.1 | 8.5 / 19.1 / 59.5 | 5.7 / 13.0 / 76.2* |
| Nemotron, semg1 all (phone recipe) | 10.8 / 23.9 / 54.8 | 12.2 / 24.2 / 56.0 | 10.3 / 13.6* / 73.8* |

| semg2 | none | zero-shot Higgs | fine-tuned Higgs |
|---|---|---|---|
| Whisper full FT | 0.8 / 4.5 / 84.5 | 0.6 / 2.7 / 89.3 | 0.6 / 4.2 / 84.5 |
| Whisper enc-only | 0.8 / 4.8 / 85.7 | 1.0 / 7.6 / 81.0 | — |
| Nemotron (phone recipe) | 1.0 / 6.4 / 82.1 | 2.2 / 5.2 / 89.3 | 3.0* / 2.7 / 96.4* |
| Nemotron, fine-tuned + zero-shot Higgs (current best) | 3.0 / 2.1 / 95.2* | | |

**semg2 sentence sets (L3 and Nemotron data-size curves).** Open-vocab training transfers better to open test.
Nemotron + three Higgs synthetic sets: open test about 60 → 8% and OOV50 about 25 → 1%. Synthetic helps Nemotron far more than Whisper.

## 4. semg3 (post-laryngectomy, no usable voice)

| model (input) | test30 | OOV50 |
|---|---|---|
| Whisper encoder-only (key clicks removed v4) | 32.5 | 61.8 |
| Auto-AVSR offline A | 39.6 | 58.8 |
| Auto-AVSR offline V (lips) | 2.4 | 11.8 |
| Auto-AVSR offline AV | 0.6 | 17.1 |
| Streaming Auto-AVSR lip reading, weight avg of 3 seeds | 3.0 | 16.7 |

Video-model zoo: USR 2.0 full FT + interpolation α=0.75 gives 0.59 / 5.26. Others: Auto-AVSR LoRA + interpolation, BRAVEn, AV-HuBERT, Llama-AVSR.
**Best (Table A).** USR 2.0 fine-tuned + video augmentation + old semg1 recordings, α=0.75, 3 seeds: test30 0.59, OOV50 4.24, Rainbow 11.90.
Streaming USR 2.0: 1.28 s chunks + augxold gives test30 1.97 / OOV50 5.12; 0.64 s gives 1.78 / 6.73.
A demo runs on a 3070 server.

## 5. Stated next steps / bottlenecks
- Bottleneck: diversity of reading material and data; overlap between test and train sets.
- New material: a 10k-sentence corpus.
- Data collection: semg-new 350 fixed vs 350 open-vocab; PD revisit with phone app.
- Future: personalised phrase completion (LLM + personal memory, dialogue/command context as prior); clarify the video gain; realistic synthetic patient data.
- **Synthetic video for semg patients has failed so far**: the generators need audio as a condition, and semg3 has none.
