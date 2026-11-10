# semg3 synthetic lip video (synvid): pre-registration

Frozen design for testing whether generated semg3 lip video, driven by healthy TTS audio, lowers semg3 lip-reading
WER on new sentences. Written before any generation or training (Phase 2.0, 2026-11-06). Changes after this point are
made only as dated amendments at the end of this file. Thresholds are never loosened.

Cluster root: `/scratch/pa2753/synvid/` (code copies in `code/`, envs in `envs/`, weights in `models/`).
Roles: `/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles/` (read-only).
Scoring everywhere: `asr_aug/shared_eval/score_shared.py` normalisation (hyphen = space).

## 1. Question
Does adding generated semg3 lip video to training lower semg3 lip-reading WER on new sentences, offline and
streaming, (a) on top of the current semg3 training data (s3_train 240 + old semg1 video) and (b) on top of that plus
the real open-vocab clips (s3ov_train 262)?

## 2. Data rules
- Training material comes only from `s3_train` and `s3ov_train`. No val, test, oov, s3ov_val, s3ov_test or rainbow
  clip feeds any decision before Phase 7, except the single judge sanity check in step 1.3 (s3ov_test, zero-shot).
- Templates (the face video a generator animates) are s3_train clips only, as their 25 fps full-frame copies
  `<session>/processed/v1/_normalized_25p/<take_id>.mp4` (A1).
- Checkpoint selection, where needed, uses pooled s3_val (30) + s3ov_val (35) (A6).
- No patient or synthetic audio or video is committed to git.

## 3. Generation pipeline, per clip
1. **Text** from the generation pool (section 4).
2. **Driving audio:** Higgs v2 base (`bosonai/higgs-audio-v2-generation-3B-base`), zero-shot voice cloning from three
   healthy stock voice prompts of the Higgs repository (`en_woman`, `en_man`, `belinda`), round-robin by clip index,
   seed 1000 + index. Never a patient voice, no adapters. Code: `make_drive.py gen`.
3. **Audio check:** whisper-large-v3 zero-shot WER against the text at most 0.10 (`make_drive.py check`).
4. **Rate match and padding** (`drive_final.py`): speech span = first to last word from a CTC forced alignment of
   the text on the clip (`align_words.py audio`, hubert-large-ls960-ft). One global time-stretch factor (pitch kept,
   ffmpeg atempo) makes the median driving active sec/word equal the median real s3_train active sec/word from step
   1.4; then a per-clip factor drawn from U[0.9, 1.1] (seed 20261107). Lead-in and tail silence are drawn from the
   real s3_train per-clip lead-in and tail values measured in step 1.4 (A5).
5. **Template:** one s3_train clip at least as long as the driving clip; templates are used about equally often;
   a template is looped only if none is long enough.
6. **Lip sync** with the generator chosen in Phase 3, 25 fps output.
7. **Crop** with the rebuilt crop pipeline (`crop_pipeline.py`: ibug RetinaFace + FAN 68 landmarks, 96x96 colour
   mouth ROI, the same steps as the real semg3 training crops), validated against the real crops in step 1.4 (A2).
8. **Judge filter (full pool only):** zero-shot USR 2.0 judge WER on the crop at most the 95th percentile of
   per-clip judge WER on real s3ov_train clips. Frozen after step 1.3: **per-clip WER <= 0.9937** (Huge judge,
   262 s3ov_train clips: mean 0.2253, mean capped 0.2217, p95 0.9937).
9. **Template-leak check (full pool only):** drop the clip if the judge output has lower WER against the template's
   own sentence than against the target sentence.

Judge: USR 2.0 (github.com/ahaliassos/usr2 @ e9e9c94), high-resource VSR checkpoint (Huge if it passes step 1.3,
else Large; step 1.3 chose Huge: 18.77% on s3ov_test vs the 17.1% reference, Large 51.03%), beam 40, CTC weight 0.1, 88x88 centre crop, grayscale, mean 0.421 / std 0.165 (`judge.py`).

Active span (A3): from the FAN inner-lip landmarks of the crop pipeline, aperture = mean inner-lip vertical gap /
outer-eye-corner distance, speed = smoothed |d aperture / dt|, threshold = 0.25 x pooled 95th percentile of the speed
over real s3_train frames (frozen in `p1/thr.json` at step 1.4), span = first to last active run of at least 3 frames
(`lip_activity.py`). The same code and threshold are used on real and synthetic clips.

## 4. Generation text
- Pool B (`/scratch/pa2753/asr_aug/data/pool_B.json`) sentences with 3 to 8 words, plus 2,000 LibriSpeech LM
  sentences (openslr 11, `librispeech-lm-norm.txt.gz`) of 3 to 8 words drawn by reservoir sampling with seed
  20261106 (A4). Code: `make_text.py`, output `data/text.json`.
- Hard leak rule: no sentence equals, or is within one word edit of, any sentence of any non-train role (all
  val / test / oov / heldout / rainbow roles of every patient); exact s3_train and s3ov_train sentences are also
  dropped. Checked by `make_text.py` (both rules) and by `asr_aug/gen/check_leak.py` (exact rule).
- The text is not targeted at OOV50 or s3ov_test words.
- Result (2026-11-06): 2,913 sentences (913 pool B, 2,000 LibriSpeech LM), 0 leaks in either check. The pool is below
  the plan's 3,000 because A4 removed pool B sentences outside 3 to 8 words; the Phase 2 bar (at least 2,000 passing
  driving clips) is unchanged.

## 5. Generator gate
Candidates (A9):
- **LS0**: LatentSync 1.6 zero-shot (github.com/bytedance/LatentSync @ a229c39, weights ByteDance/LatentSync-1.6 @
  c42c7e6; Apache-2.0 code, openrail++ weights, internal research use).
- **KS0**: KeySync zero-shot (github.com/antonibigata/keysync @ 5e2503b, weights toninio19/keysync @ c26cc46;
  Apache-2.0).
- **G-LSft**: LatentSync 1.6 fine-tuned on semg3 (section 5.1).

### 5.1 G-LSft fine-tune
Training clips: the 240 s3_train 25 fps full-frame clips only. Each clip keeps its own video; its audio track is
replaced by a healthy-voice clip of the same transcript whose words are moved onto the clip's mouth motion:
1. Word timings on the real clip: CTC forced alignment of the transcript (cut into the judge's unigram1000 units) on
   the judge's video CTC posteriors of the real mouth crop (`align_words.py video`).
2. Word timings on a Higgs clip of the same transcript (same three stock voices, round-robin; must pass the whisper
   check): CTC forced alignment with hubert-large-ls960-ft (`align_words.py audio`).
3. Retime: cut points are speech start, the midpoints of the gaps between consecutive words, and speech end. Each
   audio piece between two cut points is tempo-changed (pitch kept) to the length of the matching video piece;
   lead-in and tail silence match the video; the result has exactly the clip's frame count x 640 samples
   (`align_words.py retime`). Clips whose word counts disagree, or that would need a tempo change outside 0.25x to 4x,
   are left out.
4. Mux into the 25 fps clip and run LatentSync's own 512 px affine face alignment (`glsft_prep.sbatch`).
5. Train with LatentSync's own `scripts/train_unet.py` and its stage2_512 config from the released
   `latentsync_unet.pt`: lr 1e-5 constant, batch 1, 16-frame windows, 3,000 steps, checkpoint every 500 steps; the
   last checkpoint is used (no selection on any held-out clip). Hard cap 8 GPU-hours, resumable
   (`glsft_train.sbatch`). Steps, lr, batch and GPU time are recorded in `phase3.md`.
Because G-LSft has seen every template in training, the template-leak rule is the critical check for it.

### 5.2 Part 1 (Phase 3 pilot)
100 sentences drawn from the passing driving clips with seed 20261108, on 100 distinct s3_train templates (seed
20261109; longest driving clip first, each takes the first unused template that is long enough). The same driving
audio and templates for every candidate (`pilot.py`, `pilot_gen.sbatch`). Metrics (`gate_vid.py`):

| metric | definition | pass rule |
|---|---|---|
| crop success | generated clip exists, crop written, output frames = input frames, a face on every frame | >= 98 of 100 |
| readability | judge WER capped at 1, mean over all 100 (a clip with no crop counts as 1) | <= real s3ov_train mean + 0.15 = 0.3717 |
| template leak | share of judged clips with lower judge WER vs the template's sentence than vs the target | <= 0.05 |
| rate | median active sec/word of the output crops | within 15% of real s3_train |
| sharpness | per clip median over frames of Laplacian variance (grayscale crop); median synthetic / median real s3_train | >= 0.7 |
| too easy | share of judge WER 0, synthetic vs real s3ov_train | descriptive |
| look | 20 side-by-side real vs synthetic crops per candidate | descriptive |

A candidate passes if it meets every rule; if more than one passes, the lowest readability WER is chosen. If none
passes, synthetic video is closed for semg3 unless Preyansh decides otherwise at DECISION D2.

### 5.3 Part 2 (once R0 exists)
R0 must read the pilot clips of the chosen generator at no more than 1.5x its WER on real s3ov_train clips.

## 6. Offline arms (Phase 5)
A 2x2: real open-vocab clips (no / yes) x synthetic clips (no / yes).

| arm | training data |
|---|---|
| R0 | s3_train 240 + old semg1 video, with the current recipe's video augmentation |
| R1 | R0 + s3ov_train 262 |
| S0 | R0 + the same 1,000 synthetic clips |
| S1 | R1 + the same 1,000 synthetic clips |

The current semg3 USR 2.0 fine-tune recipe, unchanged across arms (epochs, lr, augmentation, checkpoint rule, the
alpha = 0.75 weight interpolation); synthetic clips get the same augmentation. Seeds 1 to 5, the same 1,000 synthetic
clips for every seed. R0 must reproduce the reference numbers (mean OOV50 within 1.5 pp of 4.24, test30 at most 1.5)
or the study stops. Track B (public auto_avsr lip reading) only if chosen at DECISION D3.

## 7. Streaming arms (Phase 6)
The same 2x2 with the streaming USR 2.0 recipe at 1.28 s chunks; R1 and S1 also at 0.64 s. Seeds 1 to 5. Streaming
is compared only with streaming.

## 8. Analysis (fixed before Phase 4)
- Offline primaries (Holm over 2): S0 - R0 on OOV50; S1 - R1 on s3ov_test. Paired two-level bootstrap (seeds, then
  sentences), 4,000 draws, as in `asr_aug/stack/report_stack.py`. "Synthetic helps" = Holm-adjusted p < 0.05 and 95%
  CI upper bound < 0.
- Streaming primaries: the same two contrasts at 1.28 s, their own Holm family.
- Confirmation, only if a primary passes: a step-matched long control (R0-long or R1-long, seeds 1 to 5); the claim
  stands only if the synthetic arm also beats it with CI upper < 0.
- Secondary (CIs, no claims): the same contrasts on test30, the other test set and val; R1 - R0; S0 - R1; 5-seed
  soups; new-word vs seen-word errors; share of test new words present in the generation text; rainbow (descriptive).
- WER reported in % and as word errors over word count (s3_test 169, s3_oov 228, s3ov_test 341 words).

## 9. Stop conditions
Any text leak (2.1); fewer than 2,000 passing driving clips (2.3); no candidate passes the gate (D2); fewer than
1,000 clips pass the full-pool filters (4.3); R0 does not reproduce (5.1); gate part 2 fails (5.2); Phases 2 and 3
together, fine-tune included, heading over about 20 GPU-hours.

## 10. Amendments (A1 to A9, 2026-11-06, before any generation or training)
- A1 Templates are the 25 fps full-frame copies `processed/v1/_normalized_25p/<take_id>.mp4` of s3_train rows; the
  ~30 fps variable-rate raw videos are not used.
- A2 The crop step is rebuilt from this repo's `preparation/` code (ibug RetinaFace + FAN 68, 96x96 mouth ROI, colour)
  and checked against the real training crops (step 1.4: equal frame counts on 20 clips, judge WER within 2 pp).
- A3 Active span comes from the FAN inner-lip landmarks of that crop pipeline (section 3).
- A4 Generation sentences have 3 to 8 words (s3_train is 3 to 8, median 6), for both sources.
- A5 Lead-in and tail silence are drawn from the real s3_train distributions measured in step 1.4.
- A6 Validation is pooled s3_val + s3ov_val.
- A7 The external open-vocab 10k corpus whose first 350 lines are the s3ov sentences is excluded from all text.
- A8 Power note: OOV50 has 228 words, so 4.24% is about 10 word errors. Primaries unchanged.
- A9 Third gate candidate G-LSft (section 5.1); the pilot is 100 sentences x 3 candidates with the same driving audio
  and templates; gate rules unchanged; Phases 2 and 3 together, fine-tune included, stay under about 20 GPU-hours.
- A10 Phase 3b (2026-11-07, after D2 failed for all three candidates; descriptive diagnostics, D2 unchanged). G-LSft
  checkpoint-3000 on the D2 pilot templates, scored with the frozen gate table plus the v1.8 lip model as a second,
  descriptive judge; about 3 GPU-hours; stop and report if every arm stays above 0.6 judge WER.
  3b.1 audio rate: (a) normal-rate Higgs speech span, same lead-in/tail as D2; (b) the same normal-rate words with
  equal silence inserted at the gap midpoints so the clip length equals the 1.47x stretched D2 clip. Both against
  D2 G-LSft (0.839). Word timings are the audio CTC timings drive_final.py already used for the speech span.
  3b.2 own sentence: each template driven by Higgs TTS of its own transcript retimed onto the clip's word timings
  (the A9 G-LSft training pairs, 90 of the 100 templates have one); readability, rate, sharpness; leak undefined.
- A11 Text-to-mouth step 1, splice test (2026-11-09, Preyansh; plan_semg3_text_to_mouth.md). Gate G3 is frozen here
  before any clip is built or scored. CPU unit selection from real s3_train 96x96 mouth crops (no generator):
  - Source: the 240 s3_train crops of step 1.4 minus 24 held-out clips (random.Random(20261110).sample of the take_ids
    sorted; listed in t2m/heldout.txt). No val, test, oov, s3ov_test or rainbow clip is read.
  - Word spans: the A9 video CTC word positions (glsft/video_align.jsonl, one peak frame per word) cut at the midpoints
    between consecutive peaks; the first word starts at the lip-activity span start, the last ends at its end (step 1.4,
    frozen threshold). The D2 audio CTC timings are on TTS audio, not on semg3 clips, so they cannot place real words.
  - Phones: CMUdict first pronunciation, spread inside each word in proportion to a fixed average phone duration table.
  - Units: whole-word segments; diphones (phone midpoint to next phone midpoint, the first/last diphone of a word
    extended to the word edge) for words not in the inventory; 6-frame real closed-mouth windows (lead-in/tail frames
    outside the lip-activity span) at start and end.
  - Selection: Viterbi over candidates (at most 30 per slot, fixed seed). Target cost: whole word 0, exact-phone
    diphone 0.5, viseme-class diphone 1, fallback 2. Join cost at the boundary frames: mean abs gray difference (48x48,
    /255) + |aperture difference| x 5 + |mean brightness difference| / 255 x 2. Joins cross-fade 2 frames (1 inside a
    diphone word).
  - Sets: (a) the 24 held-out sentences (their own clips excluded); (b) the 100 D2 pilot sentences. In-vocabulary word
    rate reported per set.
  Gate G3, applied to set (b); set (a) gets the same table:
  - crop validity >= 0.98: clip built, every word realised, every frame 96x96 decodable;
  - readability: mean capped WER of the zero-shot USR 2.0 Huge judge (step 1.2 settings) <= 0.372;
  - sharpness: median Laplacian variance / median over the real s3_train crops >= 0.7 (gate_vid.py rule);
  - patient likeness: per clip, aperture range (p95 - p5), opening/closing speed (p95 of the smoothed aperture
    velocity x 25) and active sec/word (lip_activity.py rule) from the carried FAN aperture; the synthetic median of each
    must lie inside the real s3_train per-clip p10..p90.
  - Descriptive only: v1.8 lip model WER (cluster only), real held-out crops as its calibration; D2 G-LSft on the same
    100 sentences (Huge 0.839, v1.8 0.863) as the comparison.
  Stop rule: set (a) readability > 0.6 -> stop and report.
- A12 Text-to-mouth step 1b (2026-11-09, Preyansh): s3ov_train joins the splice inventory, so s3ov_train is no longer a
  clean real-data control for any later step. Gate G3 is unchanged and stays frozen (A11; the likeness band stays the
  real s3_train one). Inventory = the same 216 s3_train clips + s3ov_train minus 26 held-out clips
  (random.Random(20261112).sample of the sorted take_ids, t2m/step1b/heldout_s3ov.txt). s3ov_train crops are made with
  the step 1.4 crop pipeline, word positions with the same A9 video CTC alignment of each clip's transcript, lip span
  with the frozen step 1.4 threshold. Sets: (a) the same 24 s3_train held-out sentences, (b) the same 100 D2
  sentences, (c) the 26 s3ov_train held-out sentences (open vocabulary, real clips available as the reference). Same
  judges; v1.8 trained on s3ov_train, so its numbers are flagged optimistic. Stop rule: set (b) or (c) Huge
  readability > 0.6 -> report and recommend closing the synthetic track.
- A13 Targeted text-to-mouth (2026-11-10, Preyansh; plan_semg3_t2m_targeted.md, untracked). Frozen before step 0.
  - Design: DualLip-style generator, text -> phonemes -> semg3 duration model -> 96x96 grey devcrop mouth crops in a
    frozen autoencoder latent, conditioned on one reference frame; losses L1 + perceptual, frozen lip-reader CTC from a
    reader that is not the judge (Auto-AVSR LRS3 VSR or USR 2.0 Base), smoothness. Target: unseen-word errors of the
    deployed chain on the 193 in-home clips (81/1116 errors, 24 unseen-word errors of 231 unseen-word tokens).
  - Data rules (sec 2): training pool = readable labelled semg3 clips of s3_train (240), s3ov_train (262), v3.4 train
    (298, Tianyu's split) and real conversation (~116); no clip of s3_val, s3ov_val, s3_test, s3_oov, s3ov_test,
    v3.4 val/test, rainbow or any in-home clip. 30 held-out reconstruction sentences (fixed seed, spread across
    sources, t2m2/heldout30.txt) are excluded from generator training. Generation text: public conversational English,
    3-10 words, exact and normalised leak check against every eval transcript including the 193 in-home ones.
    /scratch/th3482 read-only; outputs under /scratch/pa2753/synvid/t2m2/; nothing from Tianyu's files leaves torch.
  - Gate G4 (sec 4), frozen: judge USR 2.0 Huge zero-shot (step 1.2 settings), v1.8 descriptive only.
    (a) 30 held-out reconstruction sentences: readability <= 0.372; (b) 100 new text-pool sentences: readability
    <= 0.372 and crop validity >= 0.98; (c) sharpness >= 0.7 of real, patient likeness inside the real spread (A11
    rules). Stop rule: (a) > 0.6 after the first full generator run -> stop and report.
  - Training test (sec 5), only if G4 passes and Preyansh says go: F1 recipe, R vs R+S (synthetic <= 50% of real and
    <= 500 clips), 3 seeds, chain decode. Primary: pooled in-home word errors on the 193 clips with the deployed chain,
    R+S minus R, seed-paired bootstrap; "helps" = CI upper bound < 0.
  - Budget: step 0 CPU (plus <= 0.5 GPU-h video-CTC alignment); step 1 8-12 GPU-h; G4 < 1 GPU-h.

## 11. Files
| file | step |
|---|---|
| `judge.py` | 1.2 zero-shot USR 2.0 judge |
| `crop_pipeline.py`, `compare_crops.py`, `lip_activity.py` | 1.4 crop rebuild, crop match, active span |
| `smoke_drive.py`, `smoke_gen.sbatch` | 1.5 generator smoke tests |
| `make_text.py` | 2.1 generation text |
| `make_drive.py`, `drive_final.py`, `drive.sbatch` | 2.2 / 2.3 driving audio |
| `align_words.py`, `glsft_prep.sbatch`, `glsft_train.sbatch` | G-LSft training pairs and fine-tune |
| `pilot.py`, `pilot_gen.sbatch`, `gate_vid.py` | Phase 3 pilot and gate |
| `t2m_build.py`, `t2m_build.sbatch`, `t2m_judge.sbatch`, `t2m_gate.py` | A11 text-to-mouth step 1 |
| `t2m1b_prep.sbatch`, `t2m_build.py --with-ov`, `t2m_gate.py <dir>` | A12 step 1b |
| `phase3b_audio.py`, `phase3b_arm.sbatch` | Phase 3b arms (A10); `gate_vid.py --root .../phase3b` scores them |
| `build_env_*.sbatch`, `fetch_text.sbatch`, `p1_gpu_*.sbatch`, `SUBMIT.md` | cluster setup and Phase 1 jobs |
