# Strict gate 2: pre-registered pass thresholds

Written 2026-10-05, before any strict_gate2 script was run (file sha1 is recorded in REPORT.md). No threshold below
is changed after a result is seen; a test that misses its threshold is reported as FAIL.

Notation: for one (frame j, codebook c) cell, `lsI` / `lsT` are the log-softmax over the 1026-way codebook vocabulary
of the inference-path logits (raw `audio_lm_head` output inside `model.generate`, before the delay-pattern processor,
temperature, top-k, top-p, RAS) and of the training-path logits (`higgs_ft_lib.forward_logits` on the sequence rebuilt
by `build_example`). Per cell: `meanD = mean_v |lsI - lsT|`, `maxD = max_v |lsI - lsT|`,
`dLP = |lsI[x] - lsT[x]|` for the sampled token x, `agree = argmax lsI == argmax lsT`.

## T1 — training-path vs inference-path logit parity (adapter none, ref ft_stitch5, 10 tts_val sentences)

Noise floor (measured inside the same job, before the parity numbers are computed, on the matched training sequences):
- N1 precision noise: training path in bf16 vs the same forward with the model cast to fp32.
- N2 shape noise: bf16 training path on the full sequence vs on the sequence truncated after the middle target frame
  (same causal computation, different kernel shapes).
For each statistic S in {median meanD, p99 meanD, median maxD, mean dLP} pooled over all cells of all valid clips,
`NF(S) = max(N1(S), N2(S))`; `NF(agree) = min(N1(agree), N2(agree))`.

A clip is *valid* if generation ended with the all-EOS frame before `max_new_tokens` (1000) and its generated frames
are a well-formed delay pattern (`build_delay_pattern(revert_delay_pattern(G)) == G`). T1 needs >= 6 of 10 valid
clips, otherwise T1 is INCONCLUSIVE.

T1 PASS iff all of:
1. Structure (exact, every valid clip): the generation prompt's `input_ids` equal `build_example`'s `input_ids` up to
   the first target-audio position; the reference frames used by generation equal `build_delay_pattern(encode_wave(ref))`;
   the position whose logits sampled generated frame 0 is the last prompt token (`<|audio_out_bos|>`), and generated
   frame j is predicted by training position `tgt_pos[j] - 1` for every j.
2. Matched parity, pooled over all cells: median meanD <= 3 NF, p99 meanD <= 3 NF, median maxD <= 3 NF,
   mean dLP <= 3 NF, agree >= NF(agree) - 0.02.
3. The BOS step (j = 0) and the EOS step (first frame with cb0 = EOS) satisfy, on their own cells, meanD <= 3 NF(p99 meanD).
4. Controls fail clearly. Each of (a) training sequence shifted +1 frame, (b) shifted -1 frame, (c) training sequence
   built with a different reference (zero-shot `stitch5`, other clips and text) must have median meanD >= 3x its pass
   threshold (i.e. >= 9 NF) AND agree < NF(agree) - 0.02.

## T2 — overfit then free-run (K = 4 tts_train clips, one fixed reference ft_stitch5)

Training: train_lora.py config (r16, alpha 32, dropout 0.05, AdamW lr 1e-4, wd 0.01, grad clip 1.0, cosine with 5 %
warmup over 400 steps, token-weighted loss, fp32 LoRA on the bf16 base, same target regex); one step = the 4 clips.
Stop as soon as teacher-forced CE (eval mode, the arm's own labels) < 0.5 nats/token, checked every 10 steps, max 400.
Generation: gen_ft.FTClient with the saved adapter, temperature 0.3, top_k 50, top_p 0.95, RAS 7/2, max_new_tokens 750,
seed from run_unit (seed_base 260619825). ASR: whisper-small (transformers, nemo_asr python), greedy, English.
Metrics per clip: duration ratio = generated frames / true frames; cb0 token match = max over global lag in [-5, 5]
of the fraction of true cb0 codes reproduced at the lagged position (denominator = true length); ASR consistency =
WER(ASR(generated), ASR(true codes decoded by the audio tokenizer)).

Arms: `aligned` (labels as built), `shift_m1` (labels rolled -1: predict two frames ahead, the double-shift bug),
`shift_p1` (labels rolled +1: predict the current frame, the no-shift bug), `noeos` (eos_labels False). The aligned
job also free-runs the base model (no adapter) on the same 4 prompts as the floor.

Aligned PASS iff all of: (1) CE < 0.5 reached within 400 steps; (2) all 4 clips stop before max_new_tokens and >= 3/4
have duration ratio in [0.8, 1.25]; (3) mean cb0 match >= 0.30 and >= 5x the base-model mean; (4) mean ASR-consistency
WER <= 0.5.
Each shift control *fails at inference* iff it misses at least one of (2), (3), (4). T2 PASS iff aligned passes and
both shift controls fail at inference. `noeos` is reported only (does stopping still work: count stopping clips and
duration ratios); it does not enter the T2 verdict.

## T3 — strong conditioning controls (40 tts_train clips, same selection and matched references as gate2_diag.py)

Paired per-clip deltas (variant CE - matched CE), reported for all non-EOS tokens / cb0 / first 20 frames / EOS.
A delta is *detectable* iff on the `all` metric: mean > 0, mean >= 5 SEM, and the delta is positive in >= 90 % of clips.
Variants:
- (a) target audio from a different speaker (TORGO M03 head-mic clip closest in duration; patient text and reference).
  No TORGO control speaker (FC*/MC*) exists on the cluster; M03 is a mild-dysarthric male speaker.
- (b) patient target audio and text, reference from a different speaker (k = 1/3/5 TORGO M03 clips stitched like
  make_refs, with their prompts as reference text).
- (c) target text replaced by a different tts_train sentence of the closest character length (|len ratio - 1| reported;
  word Jaccard with the true text < 0.5).

T3 PASS iff (a) and (c) are detectable. (b) is reported as a finding: "reference conditioning used" iff detectable.

## T4 — frame-level audit (cached codes, 6 tts_train clips + T1 samples)

PASS iff all exact: (i) masked labels in the target span are exactly the BOS cells; filling them with BOS and
`revert_delay_pattern` gives codes equal to `data/codes/<utt>.pt`; decoding both with the audio tokenizer gives
bit-identical waveforms (and decoding the cached codes twice is bit-identical, else determinism is reported);
(ii) reference-span labels are all ignored; (iii) for T codes, the first label of codebook c is at target frame 1 + c,
cb c's first EOS label at frame T + 1 + c, and the last label at frame T + 8; with eos_labels False no EOS label remains;
(iv) for every valid T1 sample, cb0's first EOS in the generated frames sits at frame T_gen + 1 (one frame after
cb0's last speech frame), cb c's at T_gen + 1 + c, and the rebuilt training labels put cb0's first EOS label at the same frame.
