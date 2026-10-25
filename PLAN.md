# PLAN

## 1. Plan after new patient data (2026-10-21)

Status: **proposal, nothing implemented or submitted.** Responds to the user's note that semg1 gets more data this week/month and two more patients exist, at `/scratch/th3482/Dysarthric_ASR/data/patients` (read-only for us, see memory `th3482-read-only`; all outputs go to `/scratch/pa2753`).

### 1.1 What is actually in the tree (inspected 2026-10-21 via ssh torch)

| patient | condition | session(s) present | takes | roles in `splits/combined_v5` |
|---|---|---|---|---|
| semg1 | post-stroke dysarthria (severe) | 2026-10-01_w50s350 only; old legacy/legal in `LipVideoData` | 349 used | s1_train/val/test/oov 240/30/30/49; old_legal 238/30/30; old_legacy 76/10/10 |
| semg2 | Parkinson's (milder) | 2026-10-01_w50s350; 2026-10-16_openvocab350; 2026-10-21 Convey (34 clips) | 350 + 252 + 34 | s2 240/30/30/oov50; s2ov 189/25/38; s2conv_train 26, s2conv_heldout 8 |
| semg3 | post-laryngectomy, weak voice, mouth moves normally | 2026-10-15_w50s350 (+ Rainbow, 6) | 350 + 6 | s3 240/30/30/oov50; clickduck_v4 (`s3cd_*`) is the current audio variant |

- **The "2 more patients" (semg2, semg3) are already ingested, preprocessed and split.** Our ASR work to date used semg1 only. Nothing in `_incoming/` yet, so the **new semg1 data has not arrived**.
- Same positions = same role across patients, so pooling has no sentence leakage (`w50s350_v1/README.md`). OOV50 = positions 301-350, never trained on.
- Latest split root: `splits/LATEST_combined` -> `combined_v5`. Scoring convention: hyphens count as spaces (from 2026-10-18).

### 1.2 Already closed: do not retry

- Higgs synthetic augmentation on semg1 Whisper/Nemotron and TORGO: null (`asr_aug/README.md`, `asr_aug/torgo/README.md`). No more ratios/seeds on the 40-sentence test.
- Higgs TTS finetune similarity (S/K/R/RS): settled; reference choice not worth tuning.
- Nemotron LR schedule: corrected schedules do not beat old `noam` (flat 1e-5 11.21, cosine 1e-4 10.76, old 10.61; `revoice/README.md` amendment 3).
- CarelessWhisper sweep (memory): settled at 32.29%.
- Caveat on every old number: Whisper decode-prompt mismatch and Nemotron effective LR 1e-6 (memory `pre-20261020-recipe-bugs`). Fixed Whisper recipe is **worse** on semg1 (R_lt 11.06 vs legacy 7.43).

### 1.3 In flight (finish first; they use the same quota)

Re-voicing augmentation (`asr_aug/revoice/README.md`): Whisper V1_lt vs R_lt seeds 4-10 (primary), V1 legacy vs legacy R (secondary), Vall_lt seeds 4-6, Nemotron V1 seeds 1-3. Job status unverified since the Oct 21 log. Pass/fail is pre-registered there. Rule: if it is null, the TTS-augmentation line is closed for all patients (item F below stays dropped).

### 1.4 Proposed items

Each item: tests / expected effect / cost / pass criterion. All use the same shared WER harness and splits above; checkpoints selected on val WER, not val loss.

**A. Data audit of semg2/semg3 for audio ASR (no GPU, ~1 h).**
Tests: wav exists and is non-silent for every row of s2*, s3cd_*, s2ov_*, s2conv_*; counts match the table; semg3 click-duck audio vs click-mute v2 (v2 cuts late syllables). Effect: avoids training on broken audio. Pass: all rows resolve, per-role counts match 1.1, SNR/duration summary written to `/scratch/pa2753/multi_patient/audit.md`.

**B. Zero-shot baselines per patient (Whisper large-v3, Nemotron-0.6b; ~1 GPU-hour).**
Tests: headroom per patient on val/test/OOV50 with no finetuning. Effect: none on WER; sets the floor. Pass: table of 3 patients x 2 models x {val,test,oov}. Expect semg3 (weak voice) to be the worst for audio; if zero-shot WER > ~80% there, audio-only finetuning for semg3 is flagged low-value and item E gets priority.

**C. Per-patient finetune, frozen recipe (semg2 and semg3; ~12 Whisper + 6 Nemotron runs, ~15 GPU-hours).**
Tests: does the semg1 best recipe transfer to the other two patients. Recipe: Whisper large-v3 as used for semg1 (decision Q2 below on legacy vs fixed prompt); Nemotron `noam`. Seeds 1-3 (Whisper 1-6 if budget allows). Expected effect: large drop vs zero-shot on semg2 (milder), smaller on semg3. Pass: finetuned test WER below zero-shot by more than the seed sd (R sd ~1.5 pp on 40 sentences); report val/test/OOV50 separately. Secondary: s2ov (open vocab) trained and tested on its own split.

**D. Pooled vs per-patient (the main new question; ~1-2 GPU-days).**
Tests: train on s1+s2+s3 train rows (720 clips) vs each patient alone; evaluate each patient's own test. Effect: pooling should help the data-poor, shared-vocabulary setting (same 50 words) and help OOV the most; could hurt semg1 (most severe). Pass (set before running): pooled beats per-patient on a patient only if paired sentence-level bootstrap 95% interval lies below 0 and the effect >= 3 pp (the effect size our power supports). Add a leave-one-patient-out run (train on two, test third) to measure transfer to a new patient.

**E. Video / AV on semg3 and semg2 (existing Emformer AV model; ~1-2 GPU-days).**
Tests: the original "do videos help" question, now with a patient whose audio is weak. Compare audio-only baselines (B/C) against the existing AV Emformer RNN-T and video-only on the same splits. Effect: AV > audio-only most on semg3. Pass: AV test WER below the best audio-only WER on semg3 with non-overlapping paired-bootstrap interval. Cost depends on the preprocessing status (semg3 processed v1 exists).

**F. Synthetic augmentation for semg2/semg3: dropped unless 1.3 shows a clear gain.** Would need a per-patient Higgs adapter (about 25 min each) and the clean-split rule (exclude ASR val/test from TTS training). Not planned.

**G. New semg1 data (when it lands).** Checked 2026-10-21 after the user said semg1 "has more data now": `semg1/sessions/` holds one session (2026-10-01_w50s350, last modified Oct 18); `_incoming/` is empty; nothing under `/scratch/th3482` I can read is newer than Oct 19 for semg1. So either the new session is not copied yet, or "more data" means the Oct 18 consolidation of old sessions (legacy 96 + legal 298 + oct1 349 takes) with the oct1 session. Open question 3 asks which.
Step 0 (no GPU): ingest per `README.md` "Adding a new session" (we do not own the tree, so ask th3482 or work from a read-only link); build a new split version; confirm sentence overlap with existing semg1 train/test. Step 1: **freeze the recipe and the evaluation protocol in writing before touching the new data.** Step 2: treat the new session as a fresh held-out test (different day) to check whether the old 40-sentence results generalize; train on old, test on new, then train on old+new. Effect: more reliable semg1 numbers (current test is 40 sentences, R sd 1.5 pp). Pass: report both; a gain claim needs the pre-registered interval rule from the earlier studies.

**H. Result write-back (continuous).** Append each outcome to `asr_aug/README.md`-style records and to `PLAN.md`, including negatives.

### 1.5 Suggested order

A -> B -> (in-flight re-voicing results) -> C -> D -> E; G as soon as data arrives (do the ingest audit even before training). Run-legal-first rule is old (legal split); here the first split is w50s350 test+OOV50, with the merged `combined_v5` views used only after per-patient results validate.

### 1.6 Open questions (need your answer before I start)

1. **Scope.** Is the target now per-patient models, one pooled model, or both (D covers both)? And is audio-only still the focus, or should AV (E) be in this round?
2. **Whisper recipe.** New runs on semg2/3: legacy prompt (best semg1 numbers but known mismatch) or fixed `lang_task` (correct, worse on semg1 R)? I lean towards running both on one patient first. Your call.
3. **semg1 new data.** Expected date, sentence set (same 50-word set, open vocab, or new), and where it will land (`_incoming/` under th3482, or somewhere we can read)? I will not write under th3482.
4. **semg3 audio.** Use clickduck_v4 as current, correct? Is audio-only for semg3 worth training at all, or video-first?
5. **Budget and queue.** Account and GPU budget for roughly 3-4 GPU-days; /scratch is at about 1.2 TB of 5 TB after the cleanup.
6. **Convey and Rainbow clips.** Keep them as test-only (s2conv_heldout, s3_rainbow), or also use s2conv_train in training? Default: test-only.

## 2. Decisions and queued jobs (2026-10-23)

Decisions (user): per-patient AND pooled; audio only; semg3 deferred; diagnose legacy-vs-fixed Whisper prompt first; all jobs to finish by 11:00 on 2026-10-23 (assumed).
Data: `asr_aug/multi_patient/make_labels.py` -> `/scratch/pa2753/multi_patient/{labels,root}`. s1 = semg1 oct1 session only (not split_v1 merged), s2 = semg2, pool12 = s1+s2 train (480). Val/test/oov per patient.
Queued on `h200_tandon` / `torch_pr_39_tandon_advanced` (only reachable queue; l40s_public starts after 10-24, a100_tandon 13:52 today). IDs in `/scratch/pa2753/multi_patient/jobs.txt`:
- 20692557 array: R_lt seeds 4-10 re-decoded with `--prefix sot_sot` (closes the 2x2 prompt diagnosis).
- 20692558: zero-shot Whisper large-v3, s1/s2 val/test/oov.
- 15 finetunes (3 lanes + 3 pooled lanes): s2 {lang_task, legacy} x seeds 1-3, s1 lang_task x 1-3, pool12 {lang_task, legacy} x 1-3.
Not queued: Nemotron (no time), s1 legacy (existing numbers use a different label set), semg3, AV.
Risk: if the h200_tandon queue does not start these before ~08:00, the 11:00 deadline will be missed.


## H. Old -> Oct 1 semg1, does Higgs synthetic data help? (frozen 2026-10-23, before any result)
Recipe: Whisper large-v3 full FT, lang_task prefix, lr 1e-5, 30 ep, val-WER selection on s1_val (`asr_aug/whisper_oct1.sbatch`).
Eval: s1_val (selection), s1_test, s1_oov (clean for ASR and for the Higgs adapter, trained on old data only). Best + last ckpt.
Arms (`make_mix_oct1.py`), 5 seeds each (1-5): O, N, ON; OS, NS, ONS = + round(0.5*n_real) kept synthetic (half pool A/B,
pool sentences equal to any s1 val/test/oov sentence dropped; 2 of s1_oov overlapped the pools).
Primary contrasts (paired, two-level bootstrap seeds->sentences, Holm over 3 pairs x test/oov): OS-O, NS-N, ONS-ON.
"Helps" = CI upper bound < 0 on s1_test or s1_oov WER. Secondary: O vs ON (value of new real data), best vs last.
Known limits: pool text is old-vocabulary (7 of 50 s1_train words absent); 5 seeds ~ underpowered for <3pp effects.

### H results (2026-10-24; 30/30 jobs done, seeds 1-5, `asr_aug/report_oct1.py`, runs in `/scratch/pa2753/asr_aug/runs/oct1_*`)
WER %, mean over seeds, val-selected ckpt (test / oov): O 73.7/81.3, OS 61.5/66.5, N 6.4/31.0, NS 6.3/26.5, ON 7.0/27.4, ONS 7.6/21.4.
Paired diff (CI, Holm p): OS-O oov -14.8pp [-23.4,-7.0] p=0.003 (only pass); ONS-ON oov -6.0 [-11.3,-1.1] p=0.075;
OS-O test -12.2 [-25.0,+1.4] p=0.31; NS-N oov -4.6 [-11.0,+1.6] p=0.46; NS-N test -0.1; ONS-ON test +0.6.
Last ckpt: same picture (OS-O oov -16.3 p=0.015; ONS-ON oov -4.3 [-9.2,+0.2] p=0.25).
Verdict under the frozen rule: synthetic helps only when no Oct 1 real data is present (domain gap; old model knows none of the Oct 1 sentences).
With Oct 1 train, test is ~6-7% (floor), no synthetic effect. oov: trend -4 to -6pp, not confirmed after Holm.
Caveats: 30 test / 49 oov sentences; O/OS test sd 5-9pp; pool vocab old-domain (67 of 94 oov words covered).
Open: more seeds (6-10) for ON/ONS/N/NS on oov, or synthetic text covering Oct 1 vocabulary (exclude exact val/test/oov sentences).
Extension (2026-10-24, user "continue"): seeds 6-10 for N, NS, ON, ONS submitted (20 jobs 20836615-20836635, torch_pr_39_general, 4 dependency lanes; ids in `/scratch/pa2753/asr_aug/jobs.txt`). Not pre-registered as a replacement: report seeds 1-5, 6-10 and pooled 1-10 separately; the pooled oov contrasts (NS-N, ONS-ON) are the test, same Holm rule.

### H results, 10 seeds (2026-10-25; `asr_aug/report_oct1.py` now seeds 1-10 for N/NS/ON/ONS, 1-5 for O/OS)
Val-selected ckpt, mean WER test/oov: N 7.0/30.7, NS 7.3/25.5, ON 7.3/29.7, ONS 8.3/22.0 (O 73.7/81.3, OS 61.5/66.5 unchanged, 5 seeds).
Paired diff (CI, Holm over 6): ONS-ON oov -7.7pp [-12.6,-3.3] p=0.003 (pass, upper<0); OS-O oov -14.8 p=0.003 (pass);
NS-N oov -5.2 [-9.2,-1.3] raw p=0.013, Holm p=0.050 (CI excludes 0 but Holm borderline); all test diffs null (ONS-ON +1.1, NS-N +0.3, OS-O -12.2 p=0.23).
Last ckpt: OS-O oov -16.3 Holm 0.015; ONS-ON oov -5.7 [-9.6,-1.8] Holm 0.020; NS-N oov -3.3 [-7.6,+1.1] Holm 0.42 (not sig); tests null.
Verdict: synthetic reduces oov WER with and without old real data (ONS-ON confirmed at 10 seeds; NS-N CI excludes 0 for best ckpt only); no test effect (floor ~7%).
Caveat: O/OS 5 seeds and 60-80% WER (domain gap, cause unverified); seeds 6-10 were added after seeds 1-5 looked borderline (not pre-registered as replacement).

### H2. 2-fold oov-text synthetic (pre-registered 2026-10-24, before any fold training)
Question: does synthetic speech whose *text is the oov sentence set* (caregiver-known target vocabulary) help oov WER?
Exact oov text as synthetic input would train on the eval text, so: the 49 unique s1_oov sentences are split into folds A (25) / B (24)
(`make_folds.py`, seed "oov-2fold"). Higgs adapter B_r64 ep8 (old data only) synthesizes fold X text: 4 refs x 3 seeds
(~300 / 288 clips), duration+silence filter only (`filter_fold.py`). Arms NF, ONF = real (N or O+N) + min(0.5 x n_real, pool)
fold-X clips, no pool A/B; trained on fold X, **scored only on oov rows whose sentence is in the other fold**; swap X.
Seeds 1-5 per (arm, fold). Comparators (same oov rows): N, ON (no synth) and NS, ONS (generic synth). Fold-pooled oov WER
per seed, paired two-level bootstrap (seeds, sentences); primary NF-N and ONF-ON; Holm over those 2; "helps" = CI upper < 0.
Secondary: NF-NS, ONF-ONS (does targeted text beat generic). test/val reported unconditional. Caveat: same-domain
sentences share words ("can i", "i need") across folds, which is the intended transfer being measured; 24-25 sentences/fold is small.

### H2 status (2026-10-25)
Fold clips filtered (`filter_fold.py`; fixed dedup key prompt_id->audio_path, which had collapsed 300 clips to 25): foldA 292 kept/300, foldB 279/288.
20 jobs submitted (NF/ONF x fold A/B x seeds 1-5, torch_pr_60_tandon_priority, l40s_public, ids 20919428-20919454, `/scratch/pa2753/asr_aug/jobs.txt`).
Analysis: `asr_aug/report_h2.py` (untested until runs finish; run on torch with whisper_asr python).
