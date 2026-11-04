# Item 8: SGEM test-time adaptation on the stack Whisper models (pre-registered 2026-11-03, before any TTA decode)

PLAN.md item 8. Source: Wang et al., NCMMSC 2026 (arXiv 2610.23117), which applies SGEM (Kim et al., Interspeech 2023,
arXiv 2306.01981, code github.com/drumpt/SGEM) to dysarthric AVSR on unseen CDSD speakers. That paper picks the best
step count per speaker on the test set; here the step count and parameter scope are picked on val only.

## Method (`sgem_whisper.py`)
Episodic, per utterance, no labels. For each test utterance:
1. Restore the model to the checkpoint; fresh AdamW (torch defaults otherwise) and cosine schedule over N = 10 steps.
2. Each step: beam-5 decode (lang_task prefix `<|en|><|transcribe|><|notimestamps|>`, as in `asr_aug/stack`)
   gives the pseudo-label y. Teacher-force y through the decoder to get logits o_i at every predicted position.
   - GEM: mean over the content positions (EOT excluded) of the Renyi entropy H_alpha(softmax(o_i / T)).
   - NS: mean over all predicted positions of -log(1 - sum_j [softmax(o_i)_j < tau] softmax(o_i / T)_j).
   - Loss = GEM + lambda_NS * NS; one optimizer and one scheduler step. If y is empty, the step is skipped.
3. The adapted model is decoded with beam 10 (the stack's evaluation decode) after steps 0 (= no TTA), 1, 3, 5, 10.

SGEM values: T = 2.5, tau = 0.4 / C (C = Whisper vocab, 51,866), N = 10, beam 5 for pseudo-labels, no LM.
Two parameter scopes, with the per-scope values Wang et al. report for the same scopes:
- `ln`: every LayerNorm in the encoder (2 per layer + final); lr 2e-4 -> 2e-5, alpha 1.5, lambda_NS 1.
- `feat`: the encoder conv stem (conv1, conv2; Whisper's feature extractor, frozen in our finetune); lr 4e-5 -> 2e-5,
  alpha 1.5, lambda_NS 1.
The decoder is never updated. fp32, batch 1. Repetition loops (hyp longer than 3x the ref in characters) are counted.

## Models and sets
- E1 (old split_v1 + s1_train; never saw semg2) and E2 (E1 + s2_train + Higgs) single seeds 1-5, val-selected `best/`
  checkpoints from `asr_aug/stack`. Soups are secondary.
- Sets: s1/s2 val, test, oov (30/30/30/49/30/50 sentences), the stack's labels and audio.
- E1 on semg2 is the "unseen patient" case, which is the setting of Wang et al.

## Selection (val only)
Per arm, the (scope, step) in {ln, feat} x {1, 3, 5, 10} with the lowest pooled s1_val + s2_val WER (mean over the 5
seeds) is the TTA system. Ties go to fewer steps, then `ln`. The selected system is tested even if it is not better
than step 0 on val (that is then reported). Every other cell is decoded on test/oov too, descriptive only.

## Tests (`report_tta.py`)
Paired two-level bootstrap (seeds, then sentences; 4000 draws; `report_stack.boot`), TTA minus step 0, Holm over 2:
- **Primary 1 (new words, seen patient):** E2 on s1_oov.
- **Primary 2 (unseen patient):** E1 on semg2, s2_test + s2_oov pooled (80 sentences).
- "TTA helps" = 95% CI upper bound < 0 and Holm p < 0.05.
- Secondary (CIs, no claims): E2 s1_test, s2_test, s2_oov; E1 s1_test, s1_oov; the oracle best step per set
  (the number Wang et al. report, shown to compare, never as the claim); soups; loop counts.

## Known limits
- Val is in-domain and near its floor for E2 (1-3% WER), so selection has little signal on E2.
- Sets are small (30-50 sentences); s1_oov seed sd was ~2.2 pp, so effects under ~3 pp are unlikely to resolve.
- Nemotron (RNN-T; SGEM's transducer recipe updates the encoder at lr 4e-6) is not in this design. It would be a
  separate amendment after this result.

## Compute
About 15 s per utterance (10 adapt steps + 5 eval decodes on large-v3), 219 utterances, so about 1 GPU-hour per model
and scope: 10 models x 2 scopes = ~20 GPU-hours. Runs on B200 (preemptible) in the pdis env (torch 2.8 cu128,
transformers 4.46), since the whisper_asr env has no B200 kernels. Step 0 is decoded in the same env, so it may differ
slightly from the stack's own beam-10 outputs; the comparison is always TTA vs step 0 from the same run.

## Log
- 2026-11-03: smoke 21791293 (E2_seed1, 3+3 utterances, both scopes): loads in pdis, reads the mp4 audio, both loss
  terms fall every step. Full run 21793198-202 (B200, 4 processes per GPU, 1h31-1h55 each, GPU 95-100%).
- **2026-11-04 result: null on both primaries** (`results/tta/report_tta.out`; descriptive follow-ups from
  `tta_extra.py` in `results/tta/tta_extra.out`).
  - Val: every (scope, step) cell is worse than step 0 in both arms (E1 7.69 -> 7.75-8.19; E2 1.26 -> 1.37-1.59).
    Selection therefore picked the mildest cell, `ln` step 1, for both arms.
  - Primary 1, E2 s1_oov: +0.00 pp [-4.44, +4.41], Holm 1.0. Primary 2, E1 semg2 test+oov: -0.15 pp [-4.20, +3.83],
    Holm 1.0. Neither helps. All secondaries are within +-0.3 pp of zero.
  - The pre-registered bootstrap draws seeds independently for the two systems, so its CIs are wide. A seed-paired
    bootstrap (same seed in both; descriptive) gives E2 s1_oov +0.00 [-0.53, +0.51] and E1 semg2 -0.15 [-1.06, +0.51]:
    any effect of the selected system is under about 1 pp.
  - TTA rarely changes the output: `ln` step 1 changes 0.9% (E2) / 1.8% (E1) of hypotheses; even 10 `feat` steps
    change only 5.9% / 14.5%.
  - Oracle best step on test (Wang et al.'s way of reporting; descriptive only): at most -1.3 pp (E1 s1_oov 20.81 ->
    19.55, feat 5; E1 s2_test 9.94 -> 8.99, feat 5). These cells were worse on val, so they cannot be chosen fairly.
  - One repetition loop: E2_seed1 `feat` step 10 on s1_oov ("i need help ..." for 630 characters).
  - Reading: on our finetuned large-v3, beam pseudo-labels are already confident, and entropy minimisation mostly
    reinforces the current hypothesis. Wang et al.'s large gains came from models at 50-140% CER; our models are at
    1-21% WER. Nemotron is not run (no amendment).

