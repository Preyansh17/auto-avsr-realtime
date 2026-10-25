# Strict gate 2: report (2026-10-05)

Thresholds: `THRESHOLDS.md`, sha1 `aebad715f11090d4df08685ff312daffeb9ff31b`, written 23:18 EDT before any test ran
and not changed afterwards. Raw outputs: `/scratch/pa2753/higgs_ft/strict_gate2/out/{t1,t3,t2_*}/*.json`,
logs in `strict_gate2/logs/`. Jobs: T1+T4 19261242, T3 19261245, T2 aligned 19261248, shift_p1 19261254,
shift_m1 19261251, noeos 19261257 (all L40S, l40s_public / torch_pr_60_general; wall times 2:27, 2:17, 3:10, 3:43, 2:40, 2:10 = 16 GPU-min).
A_r16 (job 19260699), its lock and its logs were not touched.

## Verdict

| test | verdict | headline numbers |
|---|---|---|
| T1 logit parity, training path vs inference path | **PASS** | matched median meanD 0.032 vs noise 0.033; shift ±1 2.86 (86× noise); other ref 0.58 (17×) |
| T2 overfit then free-run | **PASS** | aligned: CE 0.147 at step 60, 4/4 stop, duration 0.90–1.11, cb0 match 0.66 (base 0.038); both shifted-label controls fail at inference |
| T3 conditioning controls | **FAIL** (criterion (a)) | (a) other-speaker target audio −0.072 ± 0.043 on all tokens; (b) other-speaker ref +0.195 ± 0.013, 100 %; (c) length-matched text +0.029 ± 0.003, 100 % |
| T4 frame-level audit | **PASS** | labels → codes → decode bit-identical; EOS label at T+1 (cb0) matches generation on 10/10 samples |

The training sequence is the inference sequence: on 10 of bingge's own sampled utterances, the logits the trainer
computes equal, within bf16 rounding, the logits generation sampled from, at every frame including BOS and EOS, and the
one-frame-shifted and wrong-reference versions miss by 17–86× that noise. End to end, a LoRA trained with
train_lora.py's config memorises 4 clips and reproduces them in free-run through gen_ft.py (66 % of cb0 codes, correct
lengths), while the same training with labels off by one frame in either direction gets its loss under 0.5 but
generates wrong-length or wrong-content audio. T3 fails only its pre-registered criterion (a), a weakness in how I
designed that criterion (explained below), not evidence of misalignment; the reference and text conditioning both
come out clearly in (b) and (c).

**Recommendation: A_r16 is safe to keep.** No alignment, label, EOS or prompt bug was found; there is nothing to fix.

## T1: training-path vs inference-path logit parity — PASS

Setup: gen_ft.FTClient, adapter none, ref ft_stitch5, build_plan_ft default sampling (T=1.0, top-k 50, top-p 0.95,
RAS 7/2), max_new_tokens 1000, 10 distinct tts_val sentences. A hook on `audio_lm_head` recorded the raw logits of
every forward call; `processor.decode` was wrapped to capture the frames. All 10 clips valid (finished, BOS first,
well-formed delay pattern); 14 072 (frame, codebook) cells.

Structure (exact on 10/10): generation prompt ids == `build_example` ids up to the target audio; the prompt ends with
`<|audio_out_bos|>` (128013), whose logits sampled frame 0; generated frame j ↔ training position `tgt_pos[j]-1`;
one lm-head call per generated frame; generation's reference frames == `build_delay_pattern(encode_wave(ref))`.

| | median meanD | p99 meanD | median maxD | mean dLP | argmax agree |
|---|---|---|---|---|---|
| N1 bf16 vs fp32 | 0.0334 | 0.1109 | 0.1342 | 0.0216 | 0.970 |
| N2 full vs truncated | 0.0320 | 0.1092 | 0.1337 | 0.0221 | 0.973 |
| NF (worst of N1/N2) | 0.0334 | 0.1109 | 0.1342 | 0.0221 | 0.970 |
| threshold (3 NF; agree ≥ NF − 0.02) | ≤ 0.100 | ≤ 0.333 | ≤ 0.403 | ≤ 0.066 | ≥ 0.950 |
| **matched** | **0.0323** | **0.1236** | **0.1352** | **0.0229** | **0.974** |
| BOS step (80 cells) | 0.0005 | 0.0024 | 0.0078 | 0.0005 | 1.000 |
| EOS step (80 cells) | 0.0300 | 0.1138 | 0.1293 | 0.0225 | 0.963 |
| control shift +1 frame | 2.864 | 12.0 | 11.07 | 3.48 | 0.093 |
| control shift −1 frame | 2.865 | 12.1 | 11.07 | 3.92 | 0.092 |
| control other ref (stitch5) | 0.581 | 2.13 | 2.31 | 0.356 | 0.653 |

Mean log-prob of the sampled token: inference −3.872, training −3.873 (shift ±1: −6.93 / −7.42; other ref −4.04).
Every control clears the "fails clearly" bar (median meanD ≥ 9 NF = 0.30 and agree < 0.950).
Note: at the BOS step the base model's raw log-prob of the BOS token is −24; harmless, because BOS is forced by the
delay-pattern processor at generation and masked in training.

## T2: overfit then free-run — PASS

Setup: K = 4 tts_train clips (2–5 s, sentences disjoint from ft_stitch5's sources), fixed ref ft_stitch5,
train_lora.py config (r16/α32/dropout 0.05, AdamW 1e-4, wd 0.01, clip 1.0, cosine 5 % warmup over 400 steps,
token-weighted, fp32 LoRA over bf16 base, confirmed `lora dtypes ['torch.float32']`), stop when teacher-forced CE < 0.5.
Free-run via gen_ft.FTClient with the saved adapter, T = 0.3, top-k 50, top-p 0.95, RAS 7/2, max_new_tokens 750.
ASR consistency = WER(whisper-small on generated audio, whisper-small on the true codes decoded by the tokenizer).

| arm | CE < 0.5 at step (final CE own / vs aligned labels) | stopped | duration ratios | cb0 match (lag ±5) | ASR consistency WER | meets pass rule? |
|---|---|---|---|---|---|---|
| **aligned** | 60 (0.147 / 0.147) | 4/4 | 1.06, 0.99, 0.90, 1.11 | **0.662** | 0.479 | **yes**: all four criteria |
| base model (no adapter) | — | 4/4 | 4.51, 2.54, 0.97, 1.19 | 0.038 | 0.875 | floor |
| shift_m1 (double-shift bug) | 90 (0.305 / 11.2) | 3/4 (one ran to 742 frames) | 0.67, 1.03, 0.50, 7.14 | 0.050 | 0.958 | no: fails (2), (3), (4) |
| shift_p1 (no-shift bug) | 50 (0.196 / 14.8) | 4/4 | 0.15, 0.16, 0.49, 0.96 | 0.011 | 1.000 | no: fails (2), (3), (4) |

Aligned: cb0 match 0.66 ≥ 0.30 and 17.5× base (≥ 5× required); ASR 0.479 ≤ 0.5 (narrow: 2 clips transcribe
identically to the true codes, 1 at 0.67, 1 at 1.25). One aligned clip (4f1cc57d) drifted off the memorised codes in
free-run (cb0 0.08) while keeping its length (1.06); the other three reproduce 81–91 % of cb0 codes.

Both shifted controls drive their own loss under 0.5, but loss measured against the correct labels rises to 11–15
nats, and free-run collapses (shift_p1: 15–43-frame "you" / "Hmm" outputs; shift_m1: half-length clips or a
7× ramble). **Caution for later evaluation:** shift_m1's outputs are fluent and roughly on-text (its WER against the
target text, 0.68, is *better* than the aligned arm's 0.89, because the aligned arm reproduces the patient's
recording, which whisper also misreads, 0.63 on the real clips). Text-WER / PER of generated audio would not catch an
off-by-one label bug; only code-level reproduction did.

noeos arm (eos_labels False, report only; job 19261257): CE < 0.5 at step 60 (0.094). **Stopping still works**: 4/4 clips
end on their own, but later: duration ratios 1.28, 1.01, 1.08, 1.16 (mean 1.13, vs 0.90–1.11, mean 1.01 aligned); 3/4 in
[0.8, 1.25]; cb0 match 0.66, ASR consistency 0.33. The stop decision then comes from the base model's prior, and with
n = 4 the ~12 % overrun is suggestive, not established. Keeping EOS labels (A_r16's setting) is the right default for
bingge's ramble problem.

## T3: strong conditioning controls — FAIL (criterion (a))

40 tts_train clips, the gate2_diag selection and matched references. Paired delta = variant CE − matched CE (nats/token);
matched means: all 4.435, cb0 3.33, first 20 frames 4.08, EOS 4.75. Pass rule per variant on `all`: mean ≥ 5 SEM,
mean > 0, worse in ≥ 90 % of clips.

| variant | all | cb0 | first 20 frames | EOS | detectable? |
|---|---|---|---|---|---|
| (a) other-speaker target audio (TORGO M03, closest duration) | −0.072 ± 0.043 (z −1.7), worse 42 % | +0.233 ± 0.078, 68 % | +0.557 ± 0.106 (z 5.2), 80 % | −0.518 ± 0.098, 20 % | **no** |
| (b) other-speaker reference (k = 1/3/5 M03 clips) | +0.195 ± 0.013 (z 14.9), 100 % | +0.294 ± 0.024, 100 % | +0.471 ± 0.034, 100 % | +0.037 ± 0.047, 65 % | **yes** |
| (c) length-matched different text (char-length ratio 1.00–1.09) | +0.029 ± 0.003 (z 9.6), 100 % | +0.064 ± 0.009, 88 % | +0.015 ± 0.004, 75 % | +0.223 ± 0.027, 98 % | **yes** |

Verdict by the pre-registered rule: FAIL, because (a) is not detectable. Reading:
- (a) Under teacher forcing, a different speaker's audio is as predictable as bingge's once its own earlier frames are
  in the context (the same weakness as the original shuffled-audio control, now for a different speaker). The mismatch
  shows only where the prefix is short: first 20 frames +0.56 (z 5.2, 80 % of clips), short of the 90 % bar and on a
  metric I did not pre-register as decisive. EOS is *easier* for M03 audio (−0.52), probably because TORGO head-mic
  clips end in cleaner silence. This is a limit of teacher-forced CE as a speaker detector, not a pipeline defect;
  T1/T2 test alignment directly. No TORGO control speaker exists on the cluster; M03 (mild dysarthria) was used.
- (b) **Reference conditioning is used**, strongly: every clip gets worse with another speaker's reference, most in the
  first frames (cb0 first-20 +0.79). For bingge data generation the choice of reference matters and the adapter is
  trained to read it.
- (c) Text conditioning holds with length matched (the earlier shortest↔longest pairing was not driving it); the
  biggest effect is on EOS (+0.22, 98 %), i.e. the text sets when to stop.

## T4: frame-level audit — PASS

6 cached tts_train clips (T = 124–151 frames) with ref ft_stitch5:
- masked target labels are exactly the BOS cells; filling them with BOS and `revert_delay_pattern` gives codes equal
  to `data/codes/<utt>.pt`; tokenizer decode of both is bit-identical (6/6), and decoding is deterministic (6/6);
- reference-span labels all ignored; `labels_pos` on the target positions equals `audio_labels`; no labels elsewhere;
- first label of codebook c at target frame 1 + c, first EOS label at T + 1 + c, last label at T + 8 (6/6);
  with eos_labels False no EOS label remains (0 left, 6/6).
- 10/10 T1 generations: cb c's first EOS at frame T_gen + 1 + c (cb0's EOS one frame after its last speech frame),
  and the rebuilt training labels put cb0's first EOS label at the same frame (e.g. 01cd930e: T_gen 162, EOS frames
  163…170, 171 frames).

## Deviations from the brief

- T3 (a)/(b) use TORGO M03 (mild dysarthric) as the other speaker; there is no TORGO control speaker on the cluster.
- T1 used 10 sentences (needed ≥ 6 valid; all 10 were).
- T2 control "labels shifted by one frame" was run in both directions (−1 = double shift, +1 = no shift); both are
  required to fail for T2 PASS, as pre-registered.
- GPU use: 16 GPU-min for T1/T3/T2×4 (2–4 min each on an L40S), far inside the 2 GPU-h budget. No other
  training was started.
