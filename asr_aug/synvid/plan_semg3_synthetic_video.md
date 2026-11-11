# semg3 synthetic lip video: plan and viability (draft 2026-11-06)

Rewrites PLAN.md 3.3 item 2 ("audio-conditioned synthetic video for semg3"). Plan only: no clip has been generated,
no model trained, no job submitted. Offline is compared with offline and streaming with streaming only.

Status of inputs:
- Tianyu's USR 2.0 lip-reading code and semg3 checkpoints: **requested by Preyansh 2026-11-06, not here yet.**
  Phases 0 to 4 do not need it. Phases 5 and 6 do.
- Real open-vocab video (s3ov_train, 262 clips) as a control arm: **approved by Preyansh 2026-11-06.**

---

## 0. Paste this into Claude Code first

```
Read the plan in semg3_synthetic_video_plan.md (I will paste or attach it). Repo: auto-avsr-realtime, branch
realtime-port (my Mac checkout ~/TREX/auto-avsr-realtime). Jobs run on torch over ssh. Rules:
- /scratch/th3482 is read-only. Write everything under /scratch/pa2753/synvid/ and the code under
  asr_aug/synvid/ in the repo.
- Work phase by phase, in order. Each phase ends with a "Done when" check. If a "Stop if" fires, stop and tell me
  what happened. Do not work around it or loosen a threshold.
- At every step marked DECISION, stop and ask me.
- At the start of Phase 2, write asr_aug/synvid/README.md as the frozen pre-registration (sections 3.1 to 3.7 of
  the plan with real paths filled in) and commit it. Any later change goes in a dated "Amendments" list there.
- No val, test, oov, s3ov_test or rainbow clip may feed any decision before Phase 7, except the single sanity
  check in step 1.3.
- Preemptible partitions only with resume: every generation job writes one output file per clip and skips files
  that already exist; every training job resumes from its last checkpoint.
Start with Phase 0 and report the audit before Phase 1.
```

---

## 1. Short answer: is it viable?

**Worth a cheap, gated pilot. Not worth the full training budget until the pilot shows the generated clips look and
read like semg3.** The pilot (Phases 0 to 3, about 1 to 2 days and roughly 10 to 20 GPU-hours) does not need
Tianyu's code, so it can start now.

### Why it could work
- **semg3's problem is the voice, not the mouth.** A lip-sync model draws a typical speaking mouth, which for semg3 is
  roughly right. That is the opposite of Higgs on semg1, where the generator smoothed away the pathology the ASR had
  to learn.
- **"The generators need audio and semg3 has none" goes away.** The driving audio only sets the mouth timing, so a
  healthy TTS voice (Higgs with stock voices) can drive it. semg3's own train videos supply the face, head pose,
  lighting and camera.
- **Published evidence exists, at larger scale.** SynthVSR (CVPR 2023) added audio-driven synthetic lip video and
  improved LRS3 lip reading at every real-data size they tried: 30 h real 104.0 to 43.3 WER, 438 h 36.7 to 27.9,
  high-resource 21.2 to 18.7.
- **The errors left are on new sentences.** Generated text can cover far more words than 240 sentences built from
  50 words. Tianyu's old semg1 recordings helped for the same reason (broader vocabulary).

### Why it may not
- **Almost no offline headroom.** s3_test uses the same 30 sentences as s1_test (169 words), so 0.59 is one wrong word.
  OOV50 at 4.24 is roughly a dozen wrong words out of about 280. With 50 sentences, a gain probably has to remove
  something like a third to a half of those errors to clear seed and sentence noise (step 5.3 measures this).
- **No published result for one speaker with a few hundred clips.** SynthVSR is multi-speaker and thousands of
  hours, and training on synthetic video alone failed there (100.3 WER). LipGen (2025) got +0.2 points on LRW.
- **Lip-sync models are built to look good, not to be readable.** SynthVSR needed a lip-reading loss on its
  generator to close a large real/synthetic gap (106.8 vs 49.7 WER in their cross-domain test). Blur and teeth
  artefacts sit exactly in the mouth crop the model reads.
- **Template leakage.** Inpainting models can keep traces of the template clip's original mouth motion, which turns
  into label noise. Of the candidates, only KeySync is designed against this.
- **Our own record with synthetic audio is null** on Whisper (H, H2, E2R, the closed generator benchmark), even if
  Tianyu did see fine-tuned Higgs help Nemotron and some semg1 OOV cells.
- **Real open-vocab video already exists** (s3ov_train, 262 clips) and probably helps more per clip. For the best
  system, synthetic video has to add on top of it.

### Where the odds are better
s3ov_test (offline 6.7, streaming 10.2 to 11.4) and streaming OOV50 (5.12 at 1.28 s) have more room than offline OOV50.

**My guess (a judgment, not a measurement):** about a 1 in 4 chance of a clear offline OOV50 gain, somewhat better on
s3ov_test and streaming. The gate in Phase 3 is cheap and tells us early whether it is worth going on.

---

## 2. What depends on Tianyu's code

| phase | needs Tianyu's code? |
|---|---|
| 0 audit | no |
| 1 judge and generators | no: the public USR 2.0 repo has inference and eval code |
| 2 text and driving audio | no |
| 3 generator gate | no (part 2 of the gate waits for R0) |
| 4 full generation | no |
| 5 offline arms | yes (Track A). Fallback Track B: public auto_avsr VSR, offline only |
| 6 streaming arms | yes, no fallback planned |
| 7 report | no |

Public USR 2.0 (github.com/ahaliassos/usr2, MIT; paper "Pay Attention to CTC", ICLR 2026): inference and evaluation
only, no fine-tuning recipe. Checkpoints Base / Base+ / Large / Huge on Google Drive. Input: 96x96 grayscale mouth
crops at 25 fps (MediaPipe landmarks by default). LRS3 lip-reading WER: Large 21.5, Huge 17.6 (high-resource). Enough
for a zero-shot judge, not for reproducing Tianyu's fine-tuning.

---

## 3. Design (this becomes the frozen README)

### 3.1 Question
Does adding generated semg3 lip video to training lower semg3 lip-reading WER on new sentences, offline and streaming,
(a) on top of Tianyu's Table A training data, and (b) on top of Table A plus real open-vocab clips?

### 3.2 Generation pipeline, per clip
1. **Text:** one sentence from the generation pool (3.3).
2. **Driving audio:** Higgs zero-shot with 3 healthy stock voices, round-robin. Never a patient voice, no adapters.
3. **Audio check:** whisper-large-v3 zero-shot WER against the text must be at most 0.10. Otherwise the clip's label
   would be wrong.
4. **Rate match:** time-stretch with pitch kept (pyrubberband or sox `tempo`) by one global factor, so the median
   active sec/word of the driving audio matches semg3's real train clips, then a random factor in [0.9, 1.1] per clip.
   Pad 0.3 to 0.6 s of silence at each end, like the closed-mouth lead-in of the real clips.
5. **Template:** one s3_train clip, full-frame raw video, at least as long as the driving audio. Each template is used
   about equally often. Ping-pong the template only if no template is long enough. Never a val, test, oov, s3ov or
   rainbow clip.
6. **Lip sync:** the generator chosen in Phase 3, template plus driving audio, 25 fps.
7. **Crop:** the exact pipeline (detector, smoothing, crop size, grayscale) that made the real semg3 crops Tianyu
   trains on (found in step 0.2).
8. **Judge filter (full pool only):** zero-shot USR 2.0 WER on the crop must be at most the 95th percentile of
   per-clip judge WER on real s3ov_train clips. The threshold is frozen in the README before Phase 4.
9. **Template-leak check (full pool only):** drop the clip if the judge's output is closer (lower WER) to the
   template's own sentence than to the target sentence.

### 3.3 Generation text
- **Sources, fixed before generating:** pool B (`/scratch/pa2753/asr_aug/data/pool_B.json`, 1,000 open-vocab
  AAC-register sentences from a hand-written grammar) plus 2,000 sentences of 5 to 12 words sampled with a fixed
  seed from the LibriSpeech LM text (openslr 11, `librispeech-lm-norm.txt`). 3,000 sentences in total.
- **Leakage (hard rule):** no sentence may equal any sentence of any non-train role after normalisation, or be within
  one word edit of one (all val / test / oov / heldout / rainbow roles of s1, s2, s3, s2ov, s3ov, old_*).
  Run `asr_aug/gen/check_leak.py`, plus a new one-word-edit check. Also drop exact s3_train and s3ov_train sentences.
- **Not targeted:** the text is picked without looking at which words occur in OOV50 or s3ov_test. The new-word lists
  are used only in the Phase 7 analysis. Reason: a deployed system cannot know future words, and H2 found targeted
  text no better than generic text.
- Tianyu's 10k-sentence corpus is not used unless we can confirm the s3ov sentences were not drawn from it.

### 3.4 Generator gate
Candidates (all with open code; the weight licences are checked in step 1.4):
- **LatentSync 1.6** (ByteDance, Apache-2.0 code, 512 px faces, video-driven, about 18 GB VRAM at inference, training
  code released).
- **KeySync** (Imperial College, arXiv 2505.00497, Apache-2.0 code, built to stop template-mouth leakage; pins torch
  2.4.1 cu121, so run it on L40S / A100 / H100, not B200).
- **MuseTalk 1.5** (Tencent, MIT code, 256 px, real time): backup only, if one of the first two will not install.

**Part 1 (Phase 3).** Pilot of 100 sentences from the pool (fixed seed) on 100 distinct templates, with the same
driving audio for every generator. On the pilot crops:

| metric | how | pass rule |
|---|---|---|
| crop success | face and mouth detected on every frame | at least 98% of clips |
| readability | zero-shot USR 2.0 judge WER, capped at 1, mean | at most real s3ov_train judge WER + 15 pp |
| template leak | share of clips whose judge output is closer to the template's sentence than to the target | at most 5% |
| rate | median active sec/word on the output video | within 15% of real s3_train |
| sharpness | variance of the Laplacian on mouth crops, median synthetic / median real | at least 0.7 |
| too easy (descriptive) | share of clips with judge WER 0, synthetic vs real | no rule |
| look (descriptive) | 20 side-by-side real vs synthetic crops | no rule, for Preyansh |

"Active" = frames where the lip aperture (inner-lip landmark distance) is changing, measured the same way on real and
synthetic clips. A generator passes if it meets every rule. If more than one passes, take the lowest judge WER.

**Part 2 (step 5.2, once R0 exists).** R0, which never sees open-vocab clips, must read the pilot synthetic clips at no
more than 1.5x its WER on real s3ov_train clips. This is the old item-2 bar ("readable by the current semg3 lip
model"), made relative to real clips of the same kind of text.

### 3.5 Offline arms (Phase 5)
A 2x2: real open-vocab clips (no / yes) by synthetic clips (no / yes).

| arm | training data |
|---|---|
| R0 | Table A: s3_train 240 + old semg1 video, Tianyu's video augmentation |
| R1 | R0 + s3ov_train 262 (Tianyu's "both + old" setting) |
| S0 | R0 + the same 1,000 synthetic clips |
| S1 | R1 + the same 1,000 synthetic clips |

- Recipe: Tianyu's, unchanged for all four (USR 2.0 full fine-tune, his epochs, lr, augmentation, checkpoint rule,
  and the alpha = 0.75 interpolation). Synthetic clips get the same augmentation as real ones.
- Seeds 1 to 5 (Tianyu used 3; 5 for power). Same 1,000 synthetic clips in S0 and S1 for every seed.
- Checkpoint selection, if the recipe selects: pooled s3_val (+ s3ov_val if that role exists), same rule for all arms.
- **R0 must reproduce Table A:** mean OOV50 within 1.5 pp of 4.24 and test30 at most 1.5. Otherwise stop and ask.

**Track B (only if Preyansh picks it at DECISION D3):** the same 2x2 with public auto_avsr lip reading
(mpc001/auto_avsr training code, the strongest public LRS3 VSR checkpoint in its model zoo). Tianyu's Auto-AVSR lips
number is 2.4 / 11.8, so there is more headroom. Track B results are compared only within Track B, never with Table A.

### 3.6 Streaming arms (Phase 6, Track A only)
The same 2x2 with Tianyu's streaming USR 2.0 recipe at 1.28 s chunks (his best OOV50 streaming row). Secondary: R1 and
S1 also at 0.64 s. Seeds 1 to 5. Streaming numbers are compared with streaming numbers only.

### 3.7 Analysis (fixed before Phase 4)
Scoring: `asr_aug/shared_eval/score_shared.py` (hyphen = space), as in every other table.

**Offline primaries (Holm over 2):**
1. S0 minus R0 on OOV50 (the Tianyu-comparable setting).
2. S1 minus R1 on s3ov_test (all real data, open-vocab test).

Test: paired two-level bootstrap, seeds then sentences, 4,000 draws, as in `asr_aug/stack/report_stack.py`.
"Synthetic helps" = Holm-adjusted p < 0.05 and the 95% CI upper bound < 0.

**Streaming primaries:** the same two contrasts at 1.28 s, as their own Holm family.

**Confirmation (only if a primary passes):** a step-matched control, R0-long (or R1-long), trained for as many
optimizer steps as S0 (or S1), seeds 1 to 5. The claim stands only if the synthetic arm also beats the long control
with CI upper < 0. This rules out "more training steps" as the cause.

**Secondary (CIs, no claims):**
- the same contrasts on test30, on the other test set, and on val;
- R1 minus R0 (what real open-vocab clips are worth);
- S0 minus R1 (synthetic as a substitute for real open-vocab clips; the clip counts differ, 1,000 vs 262);
- 5-seed weight-average soup per arm;
- errors on new words vs seen words, where "new" means not in that arm's real training text;
- share of OOV50 and s3ov_test new words that the generation text happens to contain;
- rainbow (6 clips), descriptive only.

**Bars vs Tianyu, descriptive only:**
- offline: test30 0.59, OOV50 4.24, s3ov_test 6.7 (both sets) and 7.61 / 6.94 soup (both + old), rainbow 11.90;
- streaming 1.28 s: test30 1.97, OOV50 5.12, s3ov_test 10.21 (both + old);
- streaming 0.64 s: OOV50 6.73, s3ov_test 11.4 (both sets).

Report WER both in % and as word errors over the word count (step 0.4).

---

## 4. Steps

### Phase 0: audit (CPU only, about 2 h)
- **0.1** List every semg3 role in `/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles/`
  (s3_*, s3cd_*, s3ov_*, rainbow) with row counts. Expect s3_train 240, s3_val 30, s3_test 30, s3_oov 50,
  s3ov_train 262, s3ov_test 53, rainbow 6. Note whether s3ov_val exists.
- **0.2** For every s3_train row, find (a) the mouth-crop file that training uses and the script and detector that
  made it, and (b) the full-frame raw video (25 fps) under `.../patients/semg3/sessions/`.
  **Stop if** raw full-frame video is missing for more than 10% of s3_train rows. The generators need whole faces,
  not mouth crops.
- **0.3** On real s3_train clips: duration, active span, active sec/word, face height in pixels. Small faces will look
  blurry after generation, since the generators work on 256 to 512 px face crops.
- **0.4** Word counts of s3_test, s3_oov, s3ov_test. Turn Tianyu's numbers into word errors (for example OOV50 4.24% of
  N words). No predictions are looked at, since none exist yet.
- **Done when** `/scratch/pa2753/synvid/audit.md` has all four.

### Phase 1: tools (about half a day, about 2 GPU-h)
- **1.1** Clone usr2 into `/scratch/pa2753/synvid/code/usr2`. Download the high-resource Large and Huge lip-reading
  checkpoints. Make an env.
- **1.2** Write `asr_aug/synvid/judge.py`: crops in, hypotheses and WER out (via score_shared). Check that our crops
  match what USR expects (96x96 grayscale, 25 fps, its normalisation).
- **1.3** Sanity check, the one allowed look at a test set: zero-shot Huge on s3ov_test. Tianyu's untrained USR 2.0
  gives 17.1. **Pass** if Huge or Large is within 3 pp. **Stop if** neither is: our preprocessing differs from his.
  Huge is the judge from here on. Then run the judge on s3_train and s3ov_train (the real references for the gate).
- **1.4** Install LatentSync 1.6 and KeySync in separate envs. Smoke test: one s3_train template plus one Higgs clip
  each. Read the weight licences on the model cards (HF `ByteDance/LatentSync-1.6`, `toninio19/keysync`).
  **DECISION D1:** report the licences. If any weight licence restricts use beyond non-commercial research, ask
  before using it.
- **Done when** the judge sanity check passes, both generators produce a clip, and the licences are reported.

### Phase 2: text and driving audio (about half a day, about 3 GPU-h)
- **2.0** Write and commit `asr_aug/synvid/README.md` (the pre-registration).
- **2.1** `make_text.py`: build the 3,000-sentence pool (3.3), run both leak checks, write `data/text.json`.
  **Stop if** any leak is found.
- **2.2** `make_drive.py`: Higgs zero-shot, 3 stock voices, one clip per sentence. Reuse the existing Higgs env and
  generation scripts from asr_aug, without adapters.
- **2.3** Whisper check (at most 0.10), rate match, jitter and padding. Write `data/drive.json` (text, voice, stretch
  factor, path).
- **Done when** at least 2,000 driving clips pass.

### Phase 3: generator gate, part 1 (about half a day, about 4 GPU-h)
- **3.1** Pilot: 100 sentences x 2 generators, same driving audio, same 100 templates.
- **3.2** Crop with the real pipeline (0.2), run the judge, compute the gate table (3.4). Script `gate_vid.py`, output
  `results/synvid_gate/gate_report.md`. Also record the measured GPU time per clip.
- **3.3** Write 20 side-by-side examples per generator to `results/synvid_gate/look/`.
- **DECISION D2:** report pass or fail per generator, the chosen one, and the cost per clip. If none passes, stop.
  One optional retry, only with Preyansh's OK: fine-tune the best generator on semg3's train videos, using TTS of
  each clip's transcript warped to word timings from a CTC alignment of the judge on the real clip, then rerun the
  gate once. If even that fails, synthetic video is closed for semg3.

### Phase 4: full generation (about 1 day; 10 to 50 GPU-h, use the pilot's measured cost)
- **4.1** Generate 1,500 clips with the chosen generator, in a fixed random order.
- **4.2** Crop, judge filter (frozen threshold), template-leak check.
- **4.3** Keep the first 1,000 that pass, in that fixed order, as `data/synth_1000.csv` in the training label format.
  **Stop if** fewer than 1,000 pass. Report the count. Thresholds change only by amendment.

### Phase 5: offline arms (needs Tianyu's code; about 20 to 60 GPU-h, measured on R0 seed 1)
- **DECISION D3**, if Tianyu's code is not here when Phase 4 ends: wait for it, or run Track B.
- **5.1** Set up Tianyu's recipe. Check his semg3 rows are our role rows. Train R0, seeds 1 to 5. Check the
  reproduction (3.5). **Stop if** it does not reproduce. Update the budget from R0 seed 1's GPU time.
- **5.2** Gate part 2 (3.4): R0 on the 100 pilot clips vs R0 on real s3ov_train. **Stop if** it fails.
- **5.3** Power: bootstrap R0 seeds 1-2 against seeds 3-5 (a null contrast) to get the CI half-width you should expect
  for a difference on OOV50 and s3ov_test. Report it. The design does not change because of it.
- **5.4** R1, S0, S1, seeds 1 to 5, same recipe and rules.
- **5.5** Soups per arm (5-seed uniform weight average; apply the alpha = 0.75 interpolation the way Tianyu does).
- **5.6** Only if a primary passes: the step-matched long control (3.7).

### Phase 6: streaming arms (needs Tianyu's streaming code; about 20 to 60 GPU-h)
- **DECISION D4:** run by default if either offline primary has a point estimate below 0. Otherwise ask Preyansh.
  Streaming has more headroom, so a null offline result does not settle streaming.
- **6.1** R0s, R1s, S0s, S1s at 1.28 s, seeds 1 to 5. R1s and S1s also at 0.64 s.
- **6.2** Decode with Tianyu's streaming decoder, same scoring.

### Phase 7: report (CPU)
- **7.1** `report_vid.py`: primaries, secondaries, Tianyu bars, gate metrics. Output `results/synvid/report.md`.
- **7.2** Update PLAN.md 3.3 item 2 with the outcome, in the style of the other items.

---

## 5. Budget (rough, to be replaced by measured numbers)

| phase | GPU-h | wall clock |
|---|---|---|
| 0 audit | 0 | 2 h |
| 1 tools | about 2 | half a day |
| 2 text and audio | about 3 | half a day |
| 3 gate | about 4 | half a day |
| 4 generation | 10 to 50 | 1 day |
| 5 offline (20 runs, + 5 if confirming) | 20 to 60 | 1 to 2 days |
| 6 streaming (30 runs) | 20 to 60 | 1 to 2 days |
| total | about 60 to 180 | about a week if the queue is free |

The pilot alone (0 to 3) is about 10 GPU-h and 1 to 2 days, and decides whether the rest is worth spending.

## 6. Stop conditions, in one place
- raw full-frame video missing for more than 10% of s3_train (0.2);
- judge sanity check off by more than 3 pp (1.3);
- a restrictive weight licence without Preyansh's OK (1.4);
- any text leak (2.1);
- no generator passes the gate, after at most one approved retry (3);
- fewer than 1,000 clips pass the full filter (4.3);
- R0 does not reproduce Table A (5.1);
- R0 cannot read the synthetic clips (5.2).

## 7. Cheaper alternatives if the gate fails
- More real open-vocab semg3 video (R1 already measures what 262 clips buy).
- Other patients' real open-vocab lip video (s2ov_train, 189 clips), after a sentence-overlap check against s3ov_test.
- Both are real video, so neither has a generator domain gap.

## 8. Sources
- USR 2.0: https://github.com/ahaliassos/usr2 ; https://arxiv.org/abs/2602.19316
- SynthVSR: https://arxiv.org/abs/2303.17200
- LipGen: https://arxiv.org/abs/2501.04204
- DualLip (text-to-lip, speaker-dependent GRID): https://arxiv.org/abs/2009.05784
- LatentSync: https://github.com/bytedance/LatentSync
- KeySync: https://arxiv.org/abs/2505.00497 ; https://github.com/antonibigata/keysync
- MuseTalk: https://github.com/TMElyralab/MuseTalk
- Tianyu's numbers: `docs/investigations/tianyu_notion_2026-10-26.md` sections 4 and 6
- s3_test word count: the same 30 sentences as s1_test (169 words), since all patients read the same w50s350 positions

## 9. Amendments after Phase 0 (2026-11-06, before any generation or training)
- A1 Templates: use the 25 fps full-frame copies processed/v1/_normalized_25p/<take_id>.mp4 of s3_train rows. The ~30 fps variable-rate raw videos are not used.
- A2 Crop pipeline: rebuild the crop step from this repo's preparation/ code (ibug RetinaFace + FAN 68 landmarks, Auto-AVSR 96x96 mouth ROI, same colour handling as the training crops) and check it against Tianyu's crops before any synthetic clip is cropped (new step 1.4).
- A3 Active span: measured from the FAN inner-lip landmarks of that crop pipeline (no mediapipe). The Phase 0 frame-difference stand-in is not used for any rule.
- A4 Text length: generation sentences are 3 to 8 words (s3_train is 3 to 8, median 6). LibriSpeech LM sampling is limited to 3 to 8 words, and pool B sentences outside 3 to 8 words are dropped.
- A5 Padding: lead-in and tail silence are sampled from the real s3_train lead-in and tail distributions measured in step 1.4, replacing the fixed 0.3 to 0.6 s.
- A6 Validation: pooled s3_val (30) + s3ov_val (35) for any checkpoint selection.
- A7 Tianyu's open-vocab-v3.3 10k corpus is excluded (s3ov sentences are its lines 1 to 350).
- A8 Power note: OOV50 has 228 words, so Tianyu's 4.24% is about 10 errors (3-seed mean 9.7). Primaries unchanged.
- A9 Third gate candidate G-LSft (Preyansh 2026-11-06): LatentSync 1.6 fine-tuned with its own training code on the 240 s3_train 25 fps full-frame videos only (never val, test, oov, s3ov or rainbow). Each training video is paired with healthy-voice driving audio time-aligned to that clip's mouth motion. Fine-tune capped at 8 GPU-hours, resumable; steps, lr, batch and GPU time recorded. The template-leak rule matters most for this candidate, since it has seen the templates in training. Phase 3 pilot becomes 100 sentences x 3 candidates (LatentSync 1.6 zero-shot, KeySync zero-shot, G-LSft), same driving audio and templates, gate rules unchanged. Phases 2 and 3 together, fine-tune included, stay under about 20 GPU-hours.

## 11. Phase 3b: cheap diagnostic checks after the D2 gate failure (added 2026-11-07, Preyansh)
No candidate passed gate D2. Before any new generator or a bigger fine-tune, two cheap checks look at why G-LSft output does not read. Both use G-LSft (checkpoint-3000) on the D2 pilot's 100 templates. All arms are scored with the frozen gate table: crop success, readability under zero-shot USR Huge, leak, rate and sharpness. Each arm also reports v1.8 WER as a descriptive second judge. No val, test, oov, s3ov_test or rainbow clips are used. Total cap is about 3 GPU-hours. If every arm stays above 0.6 judge WER, stop and report. The gate rules and the D2 decision are not changed.
- 3b.1 Audio rate. Two arms on the same 100 sentence/template pairs, compared with the D2 G-LSft result (stretched audio, readability 0.839):
  - (a) Unstretched normal-rate Higgs audio, with the same lead-in and tail as D2.
  - (b) Normal-rate words with silence inserted between words, so the clip is as long as the 1.47x stretched D2 clip.
- 3b.2 Own sentence vs new sentence. Each template is driven by healthy Higgs TTS of the template's own transcript, retimed onto the real clip's word timings. The report gives readability, rate and sharpness; leak is not defined here.
