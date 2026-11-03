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

## 3. Tianyu's track vs ours, and remaining plan (rewritten 2026-10-26 after reading the whole page)
Full transcription: `docs/investigations/tianyu_notion_2026-10-26.md`. The first version of this section (2026-10-25) was
based on the top 4 images only and was wrong in places: Tianyu does run seeds (3) and intervals, has done semg2/semg3,
streaming, and a demo. Same split rows as ours (large-v3 zero-shot semg1 test30 85.2 on both).

### 3.1 Scoreboard (WER %, test30 / OOV)
| | Tianyu best | Ours best | gap |
|---|---|---|---|
| semg1 | 4.1 / 10.0 (encoder-only + old data; weight avg 4.5 / 12.4, vote 3.9 / 10.3) | 7.0 ± 2.5 / 22.0 ± 2.8 (full FT N / ONS, 10 seeds) | behind on both, ~12 pp on OOV |
| semg2 | 0.6 / 2.7 (L3 full FT + zero-shot Higgs); Nemotron + both Higgs 3.0 / 2.1 | 0.6 / 5.7 (legacy prefix) | behind ~3 pp on OOV |
| semg3 | 0.59 / 4.24 (USR 2.0 lip reading + video aug + old semg1, α=0.75) | none | not started |
| semg2 open-vocab (s2ov_test, 38) | 1.6 (Whisper enc-only + synthetic, 12-model weight avg; Nemotron + synth 5.7; untrained L3 49.6 / Nemotron 70.3) | none | not run |
| semg3 open-vocab (s3ov_test, 53) | 6.7 lip reading (both sentence sets, 3 seeds); streaming 0.64 s 11.4 | none | not started |
| streaming | semg1 9.7 / 16.4 at 1.04 s; semg3 1.97 / 5.12 at 1.28 s; Mac/iPhone demo | patient: none (merged split only) | behind |
| TORGO | real 20.6, hybrid 18.5 (Whisper-medium, single runs) | Nemotron sentences 20.05 ± 0.14 on a harder text-disjoint split | about level |

### 3.2 What Tianyu has already closed (do not redo)
encoder-only vs full FT vs LoRA, lr 5e-6 / 1e-5 / 2e-5, 30 vs 60 epochs, Adafactor, large-v2, first/last 16 layers,
add TORGO, drop old recordings, synthetic pool 512 vs 20k, zero-shot vs fine-tuned Higgs, weight averaging, voting,
data scaling (flat by 60-180 sentences), cross-patient transfer, AV-HuBERT / Auto-AVSR / Parakeet / Canary / Qwen2-Audio /
Granite zoo, semg3 audio (32.5 / 61.8, useless) vs lips (2.4 / 11.8), streaming chunk sizes.
This drops my earlier T3 (find the patient set: same split), T4 (semg3 audio baseline: done, video wins) and T5 (latency: done).

### 3.3 What is left that Tianyu has not done (ordered)
Re-read of the page 2026-10-28: open-vocab test results added for semg2 (s2ov) and semg3 (s3ov), see the transcription section 6.
Each one is grounded in a gap the page states or a result it shows.
0. **Benchmark nvidia/nemotron-labs-unified-asr-0.6b (user request 2026-10-28; run first, ~2 GPU-h).** Released on
   HF 2026-10-27: Unified FastConformer-RNNT, 600M, 25 languages incl. English, one checkpoint for offline and buffered
   streaming (latency = chunk + right context, 2.08 s down to 0.16 s), language-ID prompt (`target_lang="en"`),
   punctuation and capitals. NVIDIA ships inference only so far; the card names NeMo Speech 3.1.0.
   - Step 1, zero-shot: offline, and streaming at 1.12 s (5.6/0.56/0.56) and 0.56 s (5.6/0.16/0.40), on s1/s2
     val/test/oov (shared role rows, `score_shared.py` normaliser). References: zero-shot large-v3 (s1_test ~85%) and
     zero-shot nemotron-speech-streaming-en-0.6b (our current Nemotron base), same rows.
   - Step 2, only if step 1 runs: try fine-tuning on s1+s2 train with our Nemotron recipe. This needs a NeMo training
     path for the unified model, which the card does not provide.
   - Also note: NeMo Speech 3.0 adds GPU phrase boosting for RNNT/TDT, a biasing method we have not tried (item 1).
   - Env: new `/scratch/pa2753/envs/nemo3` (built on a CPU node); `nemo_asr` (NeMo 2.7.3) is left untouched.
   - 2026-10-28: release 3.0.0 cannot pass the language prompt to its dataloader ("Unknown prompt key: 'None'"), so the
     benchmark runs on NeMo main = 3.1.0 (`envs/nemo31`, user-approved) on a uv-managed Python (the system 3.12 lacks
     Python.h, which Triton needs). Added `s2ov_test`, Tianyu's semg2 test-open, for a direct zero-shot comparison.
   - **Step 1 result (2026-10-28, zero-shot, jobs 21290573/21290574, NeMo 3.1.0+040574f, `target_lang=en`;
     hyps in `/scratch/pa2753/unified/out/`).** WER %:
     | model / mode | s1_test | s1_oov | s2_test | s2_oov | s2ov_test |
     |---|---|---|---|---|---|
     | unified, offline | 97.6 | 100.0 | 33.7 | 45.6 | 68.4 |
     | unified, streaming 1.12 s | 96.5 | 98.7 | 36.1 | 45.2 | 75.1 |
     | unified, streaming 0.56 s | 98.8 | 99.1 | 56.2 | 55.7 | 81.0 |
     | nemotron-speech-streaming-en, offline | 97.6 | 100.0 | 42.0 | 45.2 | 73.8 |
     | Tianyu, untrained (s2ov_test only) | | | | | L3 49.6, Parakeet 52.4, Nemotron 70.3 |
     - semg1: both NeMo models output nothing for most clips (unified: 27/30 s1_test and 49/49 s1_oov empty). The audio
       is fine (16 kHz mono, similar level to s2), so this is the RNNT models emitting blanks on severe dysarthria.
     - semg2: unified beats the old Nemotron offline on s2_test (33.7 vs 42.0) and s2ov_test (68.4 vs 73.8). It is
       still far from usable zero-shot, and worse than Tianyu's untrained Whisper L3 on s2ov (49.6).
     - Our old-Nemotron s2ov_test 73.8 vs Tianyu's untrained Nemotron 70.3: close; the gap fits a different normaliser.
     - Verdict: zero-shot it is not a better base for semg1. Step 2 (fine-tune) is the only way it could matter; the
       card gives no training recipe.
   - **Step 2 design (pre-registered 2026-10-28, before any fine-tune result).** NeMo 3.1 does have a training path
     (`EncDecRNNTBPEModelWithPrompt` + Lhotse prompt-index dataset, `lang_field: source_lang`), so we fine-tune with
     `asr_baselines/nemotron_finetune.py` unchanged. Code: `asr_aug/unified/{make_ft_manifests.py,ft.sbatch}`.
     - Arms: U = unified, N = old `nemotron-speech-streaming-en-0.6b`. Both trained in the nemo31 env, so only the base differs.
     - Data: the E2 real data (old_legacy + old_legal 314, s1_train 240, s2_train 240 = 794 clips; `source_lang=en`).
       Val = s1_val + s2_val (60), used for best-WER checkpoint selection only.
     - Recipe, same for both: full fine-tune, batch 2 x accum 2, warmup_flat lr 1e-5 (200 warmup steps), 60 epochs,
       speed perturb, SpecAugment defaults, seeds 1-3.
     - Eval: offline on s1/s2 test, oov, and s2ov_test. U also gets streaming at 1.12 s as a secondary.
     - Primary read: the seed mean of U vs N on s1_test, s1_oov, s2_test, s2_oov. "U better" needs all 3 U seeds below the
       N seed mean on a set, with the mean gap over 2 pp. Otherwise call it a tie. Context bars: the Whisper E2 soup and
       Tianyu's numbers (3.1). We do not expect an RNNT base to beat the Whisper encoder-only recipe on semg1.
     - **Amendment 1 (2026-10-28, after 1-epoch smoke runs on 80 clips only, before any full run):** add
       `--freeze-bn-stats` to both arms. The smoke run without it wrecked unified (s2_test 33.7 -> 71.0, many empty hyps)
       while barely moving the weights (median rel change 1e-4); conv-module BatchNorm running stats had drifted 70-150%
       at batch 2. With frozen BN stats the same smoke run gives s2_test 20.7 (unified) and 36.7 (old Nemotron).
       Env fixes found by the smoke runs: nemo31 needs `numba-cuda` 0.30.4, numpy pinned to 2.3.5 (numba-cuda uses
       `np.row_stack`), and the env's own nvJitLink 13.4 preloaded (the system 13.3 one is "outdated" for NVVM 13.4).
       The step 1 zero-shot numbers were made before the numpy pin; not re-run.
     - **Amendment 2 (2026-10-29, no full-run result read):** HPC auto-cancelled unified seed 1 after 2 h for 46.5%
       mean GPU utilisation (cancel threshold 50%); it was at epoch 36, no eval had run. Both arms rerun as batch 4 x
       accum 1 (same effective batch 4 and optimizer-step count; BN stats are frozen, so the per-step batch only
       changes gradient noise), 8 CPUs and 8 loader workers. All six runs restarted from scratch.
     - **Amendment 3 (2026-10-29, no full-run result read):** batch 4 still sampled at 42% mean GPU (36 samples over
       3 min, L40S, epoch 6), and the six single-seed jobs queued behind the per-user GPU cap. Recipe unchanged; the
       three seeds of each base now train side by side on one GPU (`ft.sbatch SEEDS="1 2 3"`, 16 CPUs split across
       seeds, starts staggered 90 s). Jobs 21322988 (unified) and 21322989 (nemotron_en); the earlier six were cancelled
       and their run dirs deleted.
     - **Preemptible GPUs (2026-10-29, infrastructure only, no recipe change):** HPC docs: `--comment="preemption=yes;
       [preemption_partitions_only=yes;]requeue=true"`, no `-p`; preemptible after 30 min, requeued in place. B200 only
       works from the nemo31 env (sm_100); whisper_asr (torch 2.5.1) has no B200 kernels, so Whisper jobs add
       `--constraint="l40s|a100|h100|h200"`. Both trainers now resume a requeued run: Whisper `--resume` (newest
       complete HF checkpoint; torch.load guard lifted for own checkpoints) and Nemotron `--resume-ckpt auto` (rolling
       .ckpt every 2 epochs + an RNGState callback; atomic best .nemo); launchers append logs and skip training after
       `train_done`. Kill-and-resume test (`asr_aug/unified/test_resume.sbatch`, jobs 21376761 / 21382997, A100): the
       resumed run is as close to an uninterrupted one as two uninterrupted runs are to each other (Whisper best
       weights max |diff| 1.73e-4 vs 1.78e-4; Nemotron final 1.34e-3 vs 1.40e-3). Bit-exact resume is not possible
       (two uninterrupted runs already differ), so a preempted seed is an equivalent, not identical, continuation.
   - **Step 2 result (2026-10-29, jobs 21322988 / 21322989, offline, val-selected best .nemo, seeds 1-3).** WER %,
     seed mean (seeds):

     | set | U = unified | N = old nemotron_en | rule (all U < N mean, gap > 2 pp) |
     |---|---|---|---|
     | s1_test | **6.31** (6.51, 5.92, 6.51) | 9.86 (8.88, 10.65, 10.06) | **U better** (gap 3.55) |
     | s1_oov | 27.21 (26.91, 28.25, 26.46) | 28.85 (26.91, 30.04, 29.60) | tie (gap 1.64) |
     | s2_test | 0.59 (0.59 x3) | 0.79 (1.18, 0.59, 0.59) | tie (gap 0.20) |
     | s2_oov | **4.68** (3.95, 5.70, 4.39) | 7.31 (7.89, 5.26, 8.77) | **U better** (gap 2.63) |
     | s2ov_test (secondary) | 43.32 (43.88, 42.19, 43.88) | 43.74 (43.04, 41.77, 46.41) | tie |

     - Unified is the better RNNT base: it wins s1_test and s2_oov and ties the rest. Fine-tuning takes it from 97.6 to
       6.3 on s1_test (zero-shot, step 1).
     - Against Whisper (context only, not like-for-like: Whisper E2 also trains on 397 Higgs clips). Unified single seeds
       beat Whisper E2 single seeds on s1_test (6.31 vs 7.57), but are far behind on s1_oov (27.2 vs 15.2; E2 soup 8.07).
       Unseen semg1 words remain the RNNT weakness. On semg2 the two are close (s2_oov 4.68 vs 2.63; E2 soup 1.75).
     - s2ov_test 43 vs 68 zero-shot. Tianyu's 1.6 trains on s2ov_train, so it is not comparable.
     - Unified streaming at 1.12 s (secondary; crashed in-job because ft.sbatch did not pass the NeMo 3.1 checkout,
       fixed and rerun as job 21388126 on B200). Seed means vs offline: s1_test 8.88 vs 6.31, s1_oov 27.80 vs 27.21,
       s2_test 0.59 vs 0.59, s2_oov 5.85 vs 4.68, s2ov_test 43.88 vs 43.32. Streaming costs about 2.6 pp on semg1 test
       and 1.2 pp on semg2 oov, and is about even elsewhere. NeMo fell back from full CUDA graphs on B200 (warning
       only), which does not change the decoding result.
   - **Step 3 = v2, semg2 phone target (pre-registered 2026-10-29, user-approved, before any v2 result).** Bar is
     Tianyu's semg2 package `semg2_best_v1_20261029` (extracted without weights to `/scratch/pa2753/tianyu_semg2_best_v1`;
     his `tools/score.py --verify` = ALL MATCH). Phone bars, WER %: test-open38 8.13 (his Nemotron-en 3-seed soup, 80 ms)
     / 5.69 (1.04 s); conv34 5.88 (current App model, 80 ms); offline context: his Whisper 2.03 / 2.71.
     - Blocker: all three of his Higgs pools (5,378 clips under `runs/semg_oct1_likeness2/pool_v{3,5}`) are unreadable to
       us. So v2 starts real-only (v2r); a Higgs arm needs either his read access or our own semg2 Higgs pool (user's call).
     - Data v2r: s2ov_train + s2_train (189 + 240 = 429), val s2ov_val + s2_val (55); role files checked identical to his
       package CSVs. Recipe = step 2 unchanged (frozen BN, warmup_flat 1e-5, 60 epochs, batch 4, seeds 1-3, packed).
       Arms U2r (unified) and N2r (nemotron_en), jobs 21379643 / 21379644. Zero-shot rows on his sets: job 21379642.
     - Eval: `tianyu_eval.py` / `teval.sh`, his CSVs and scorer, sets val25, val30, test30, oov50, test-open38, conv34.
       U: offline + buffered streaming at 1.12 / 0.56 / 0.16 s latency (C0.56R0.56, C0.16R0.40, C0.08R0.08).
       N: offline + cache-aware 80 ms (70,1) and 1.04 s (70,13), his phone modes. NeMo 3.1 scripts, not his tracer, so
       streaming rows are close to but not identical with his pipeline; N2r at 80 ms is the in-pipeline reference.
     - Primary read: U2r vs N2r seed mean on test-open38 and conv34, U at 0.16 s vs N at 80 ms (and offline vs offline).
       Same "all 3 seeds below the other arm's mean, mean gap > 2 pp" rule as step 2. Beating Tianyu's bar is secondary
       and descriptive (his arm has Higgs data and a 3-seed soup, ours not yet); 1 word = 0.4 pp on test-open.
     - **Amendment 1 (2026-10-29, no v2 result read):** Tianyu granted read access to his Higgs pools; user asked to try his
       clips and ours. Arms, all on B200 (preemptible, resume verified by job 21376761), 3 seeds packed per job:
       U2r / N2r (real 429, 60 epochs; replace the L40S jobs 21379643/44) = 21383152 / 21383153;
       U2h / N2h (real + his 5,378 Higgs rows from the package train.csv, 11 epochs = his epoch count, ~16k steps) =
       21383154 / 21383156; U2o (real + our 1,751 kept semg2 Higgs clips from the stack study, 30 epochs, ~16k steps;
       text checked free of every eval sentence) = 21383157. No N2o: only unified is the candidate.
       2026-10-30: B200 queue estimated start Oct 31 16:33, so all five were cancelled before starting and resubmitted
       (same recipes, preemptible) by `--test-only` start estimate: U2r 21410746 / N2r 21410747 on H100; U2h 21410748 /
       N2h 21410749 / U2o 21410750 on L40S. Each compared pair shares a GPU type; e2 baselines also ran on L40S.
       2026-10-30 13:30: preemptible partitions preempt every ~30 min. A v2h epoch (4,356 steps) took 22 min (N2h) to
       ~37 min (U2h) with 3 seeds packed on L40S, so with CKPT_EVERY=2 neither ever wrote a resume .ckpt and every
       requeue restarted from scratch (8 restarts). Cancelled; dirs kept as *_v2h_seed*_preempted_20261030. Resubmitted
       one seed per job on H100 with CKPT_EVERY=1: U2h 21437279/21437281/21437284, N2h 21437280/21437283/21437285
       (seeds 1/2/3). U2h vs U2r is now same-GPU (H100); U2h vs U2o is H100 vs L40S (descriptive read anyway).
       N2r done (H100, 3 seeds): offline / 1.04 s look-ahead WER test30 0.59, oov50 1.46, test_open38 7.73,
       conv34 5.73; 80 ms look-ahead 1.18 / 1.61 / 9.76 / 11.46 (Tianyu phone 80 ms: 1.78 / 1.32 / 8.13 / 9.05).
       2026-11-01 results (3-seed means, WER test30 / oov50 / test_open38 / conv34; U2o seed 1 only, seeds 2-3 running):
         offline:  N2r 0.59/1.46/7.73/5.73  N2h 0.59/1.61/6.77/7.54  U2r 0.59/2.19/7.86/7.39  U2h 0.59/0.44/6.23/3.62
                   U2o(s1) 0.59/0.44/9.35/6.33
         ~80 ms:   N2r cache70_1 1.18/1.61/9.76/11.46  N2h 0.99/2.34/9.49/12.97 (conv sd 5.0)
                   U2r C0.08/R0.08 3.55/6.43/34.4/32.3  U2h 14.6/18.1/57.1/76.5  U2o(s1) 10.7/13.6/39.8/52.9
         ~1 s:     N2r/N2h cache70_13 = offline (70,13 is the model default)  U2r C0.56/R0.56 0.79/2.92/14.8/9.80
                   U2h 0.79/0.88/14.1/9.50
       Reads: (1) best offline = U2h (his synthetic helps unified offline: conv 7.39 -> 3.62, oov 2.19 -> 0.44), still
       behind Tianyu offline Whisper on test-open (6.23 vs 2.03). (2) unified does not stream well after our fine-tune:
       at ~1 s it is worse than old Nemotron at 1.04 s, and at 80 ms it collapses, worse with synthetic data. Our
       fine-tune trains at the full default context only (no --att-context-size / chunk randomisation), so low-latency
       modes are never trained. (3) best 80 ms system = N2r (old Nemotron real-only), within ~1.6-2.4 pts of Tianyu.
       Next candidate: streaming-aware fine-tune (multi-lookahead att_context sampling for nemotron_en; chunked
       training for unified).
     - **Step 4 (v3) pre-registration (2026-11-01, user: "Yes, plan it"; no v3 result read).** Diagnosis:
       (a) nemotron_en is already trained multi-lookahead: the checkpoint's att_context_size lists [70,13],[70,6],
       [70,1],[70,0] and the encoder samples one per training batch (uniform), exactly Tianyu's
       --nemo-att-context-probs 0.25x4. Its 80 ms gap to him is not a missing streaming objective.
       (b) unified is NOT: its att_chunk_context_size ([70] / [1,2,7,13] / [0,1,2,3,4,7,13]) and
       streaming_loss_weight 0.5 are stored in the config but never read by NeMo's training code (checkout 040574f),
       so every unified fine-tune so far trained full-context only. Explains its streaming collapse, worse with
       more steps (v2h).
       (c) Tianyu's phone model is a uniform weight soup of 3 seeds (7/1/42) of his recipe (package
       recipe/soup_info.json), not a single run; his seeds also use batch 2 x accum 2, cosine lr, wd 1e-3, his three
       Higgs pools at 50% sampling ratio.
       Arms:
       S (no training): uniform 3-seed soups (asr_aug/unified/soup.py) of N2r, N2h, U2r, U2h (U2o when done), scored
         with teval.sh. Read: soup vs its 3-seed mean, per mode.
       U3r / U3h: unified, recipe v2r (60 ep) / v2h (11 ep), --unified-stream-prob 0.5 (per batch: 50% full context,
         50% [70, chunk, right] uniform over the checkpoint's own lists), 3 seeds, one seed per H100 job, then soup.
         Everything else as U2r / U2h. Checkpoint selection stays on full-context val WER.
       teval.sh adds unified C0.08 R0 (80 ms, matching Tianyu's 80 ms look-ahead) to its modes.
       Reads (pre-registered): primary = U3 vs U2 (same recipe) at C0.08/R0, C0.08/R0.08, C0.56/R0.56 on
       test_open38 and conv34, 3-seed means; offline must not degrade by more than 1 WER pt on either. Unified
       becomes the low-latency candidate only if U3 (or its soup) beats N2r / N2r-soup at matched latency
       (unified C0.08/R0 vs nemotron_en cache70_1). Single seeds are not read (seed sd up to 5 pts at 80 ms).
       Jobs: smoke 21607107 (L40S); soups 21607108 (A100); U3r seeds 1/2/3 = 21607111/21607114/21607120,
       U3h = 21607113/21607118/21607123 (H100, afterok on the smoke).
       Update 2026-11-01 20:10: smoke training passed (log prints the streaming-training line; [best] saved) and its
       teval decoded every mode incl. C0.08/R0 before a preemption, so the smoke was cancelled and the U3 jobs were
       resubmitted without the dependency and on "a100|l40s|h100" (H100 estimate was Nov 2-3): U3r 21611021/21611023/
       21611026, U3h 21611022/21611025/21611027. Fix: a requeued soup job skipped teval when any scores.json existed,
       leaving modes missing; tianyu_eval.py now skips per mode instead, and soup2 21611000 (SEEDS_TOO=1) fills
       C0.08/R0 for the U2r/U2h soups and their seeds.
       Arm S results (soup, WER %; test30 / oov50 / test_open38 / conv34; 3-seed mean in brackets for test_open38 / conv34):
         N2r offline 0.59 / 1.75 / 8.13 / 4.98 [7.73 / 5.73]; cache70_1 0.59 / 2.63 / 8.94 / 9.50 [9.76 / 11.46].
         N2h offline 0.59 / 0.88 / 5.69 / 6.33 [6.77 / 7.54]; cache70_1 0.59 / 2.19 / 10.16 / 9.95 [9.49 / 12.97].
         U2r offline 0.59 / 1.75 / 7.32 / 6.79 [7.86 / 7.39]; C0.56/R0.56 1.18 / 2.63 / 14.63 / 10.41 [14.8 / 9.80];
           C0.08/R0.08 3.55 / 4.82 / 30.49 / 31.67 [34.4 / 32.3].
         U2h offline 0.59 / 0.44 / 5.28 / 2.71 [6.23 / 3.62]; C0.56/R0.56 0.59 / 0.88 / 12.60 / 9.95 [14.1 / 9.50];
           C0.08/R0.08 19.53 / 14.91 / 56.10 / 71.49 [57.1 / 76.5].
         C0.08/R0 (80 ms, soup2 21611000): U2r soup 4.73 / 9.21 / 36.59 / 33.48 [36.45 / 34.84];
           U2h soup 5.33 / 12.28 / 46.34 / 71.95 [47.29 / 68.18]. These 3-seed means are the U2 side of the U3 read.
       Read: the soup helps offline and at 80 ms cache-aware on conv34 (N2r 9.50 vs 11.46; N2h 9.95 vs 12.97), mixed on
       test_open38. Descriptive vs Tianyu's 80 ms phone bar (8.13 / 9.05): N2r soup cache70_1 is behind on test_open38 by
       0.81 (about 2 words) and on conv34 by 0.45. U2h soup offline (5.28 / 2.71) beats his phone 1.04 s (5.69 / 8.14)
       and ties his offline Whisper on conv34 (2.71). Unified at 80-160 ms remains far off until U3 (U2r C0.08/R0 conv34 34.8 vs N2r cache70_1 11.5), and the
       soup does not rescue it there. U3 started 2026-11-01 ~22:15 (U3r seeds 1/2, U3h seed 1 running; rest queued).
       **U3 result (2026-11-02, all 6 seeds done; 3-seed means, test_open38 / conv34, U2 mean in brackets):**
         U3r offline 6.37 / 5.13 [7.86 / 7.39]; C0.56/R0.56 13.68 / 11.61 [14.8 / 9.80];
           C0.08/R0.08 25.88 / 25.19 [34.4 / 32.3]; C0.08/R0 27.24 / 22.77 [36.45 / 34.84]; C0.16/R0.4 17.21 / 11.91.
         U3h offline 6.10 / 3.77 [6.23 / 3.62]; C0.56/R0.56 11.79 / 8.90 [14.1 / 9.50];
           C0.08/R0.08 65.85 / 71.95 [57.1 / 76.5]; C0.08/R0 45.12 / 50.98 [47.29 / 68.18] (seed sd 10-21 at 80-160 ms;
           seed 3 collapses: 56.9 / 69.2 at C0.08/R0).
       Reads (pre-registered): streaming training helps U3r at 80-160 ms (-8.5 to -12.1 pts, every gap over 2 pp, seed sd
       under 3) and offline does not degrade (it improves: 6.37 / 5.13). At 1.12 s it is a tie (-1.1 / +1.8). U3h is
       unstable at 80-160 ms (gaps run both ways, sd up to 21), tie offline. Gate: unified does NOT become the
       low-latency candidate: U3r C0.08/R0 27.24 / 22.77 vs N2r cache70_1 9.76 / 11.46 (soup 8.94 / 9.50), about 2x
       worse at matched 80 ms. Soup of the U3 arms (soup3 21652499) pending, per the pre-registration ("U3 or its
       soup"); a soup is unlikely to close a 2x gap. Low-latency candidate stays nemotron_en (N2r soup).
       soup3 21652499 (U3 soups, test_open38 / conv34): U3r soup offline 6.50 / 5.88; C0.56/R0.56 14.23 / 10.41;
         C0.08/R0.08 24.80 / 23.53; C0.08/R0 27.64 / 19.91. U3h soup offline **4.07 / 2.71**; C0.56/R0.56 10.16 / 9.50;
         C0.16/R0.4 15.45 / 11.76; C0.08/R0.08 64.23 / 72.85; C0.08/R0 39.02 / 48.42. Gate unchanged (U3r soup 80 ms
         27.64 / 19.91 vs N2r soup 8.94 / 9.50). U3h soup offline is our best offline model (Tianyu offline Whisper
         2.03 / 2.71: behind on test_open38, tie on conv34).
       U2o done 2026-11-02 (3-seed means, test_open38 / conv34): offline 8.13 / 6.18; C0.56/R0.56 10.98 / 8.14;
         C0.16/R0.4 14.90 / 13.88; C0.08/R0.08 40.65 / 48.87. Synthetic effect, our pool (U2o vs U2r): ties offline
         (+0.27 / -1.21), better at C0.56/R0.56 on test_open38 (-3.8), tie on conv34 (-1.66). His vs our pool (U2h vs U2o,
         descriptive): his better offline (6.23 / 3.62 vs 8.13 / 6.18). U2o soup (soup4 21661560, test30 / oov50 /
         test_open38 / conv34): offline 0.59 / 0.44 / 6.91 / 6.33; C0.56/R0.56 0.59 / 0.88 / 10.16 / 9.05; C0.16/R0.4
         1.18 / 2.19 / 13.01 / 14.03; C0.08/R0.08 8.88 / 12.72 / 39.02 / 52.04; C0.08/R0 5.33 / 7.46 / 36.59 / 39.82.
         Soup vs its seed mean: better on test_open38, worse on conv34 (offline -1.2 / +0.15, C0.56 -0.8 / +0.9); best-per-lag table unchanged.
       Best per latency vs Tianyu (descriptive, one soup each): 80 ms N2r soup cache70_1 8.94 / 9.50 vs his 8.13 / 9.05
         (behind 0.81 / 0.45); 1.04 s N2h soup cache70_13 5.69 / 6.33 vs his 5.69 / 8.14 (tie / ahead);
         offline U3h soup 4.07 / 2.71.
       560 ms lag (cache70_6, 480 ms look-ahead; job cache706 21667115, asr_aug/unified/cache706.sbatch), test30 / oov50 /
         test_open38 / conv34: N2r soup 0.59 / 1.75 / 7.32 / 6.33 (seed mean 0.59 / 1.75 / 8.67 / 6.63);
         N2h soup 0.59 / 2.19 / 8.94 / 8.14 (seed mean 0.59 / 2.92 / 7.86 / 8.90). Unified at 560 ms (C0.16/R0.4)
         U3h soup 15.45 / 11.76 on test_open38 / conv34. No Tianyu number at this lag.
       Reads: synthetic effect = U2h vs U2r and U2o vs U2r (same rule as above); his vs our pool = U2h vs U2o, descriptive
       (pool size, text domain and epochs all differ). Base comparison with his data = U2h vs N2h; N2h is also our
       in-pipeline rerun of his phone recipe (no soup, our schedule), so N2h vs his 8.13 checks the pipeline.
     - **Step 5 pre-registration (2026-11-03, user: "continue"; no Step 5 result read).** Target: the 160 ms gap
       (N2r soup cache70_1 8.94 / 9.50, oov50 2.63 vs Tianyu 8.13 / 9.05 / 1.32). Two decode-only arms, nemotron_en only.
       S6: one uniform soup of all 6 nemotron_en seeds (N2r 1-3 + N2h 1-3, same base). Scored offline, cache70_1,
         cache70_6, cache70_13. Read, descriptive (one model): S6 vs the N2r and N2h soups per lag.
       PB: NeMo GPU phrase boosting (rnnt_decoding.greedy.boosting_tree, decode-time only) at cache70_1 on the N2r soup
         and N2r seeds 1-3. Phrase lists (asr_aug/unified/make_boost_lists.py), reported apart:
           clean (the claim): the unique sentences of our real train set (s2ov_train + s2_train, 429 clips), i.e. text
             the model already trains on; no eval text.
           words (ceiling): clean + the eval-set words absent from the train vocabulary ("a caregiver supplies the
             vocabulary"); test words are known, so never the claim, and never compared with Tianyu.
         alpha in {0.5, 1, 2, 4}, picked per list by lowest pooled val25 + val30 WER on the N2r soup; other alphas
         descriptive. Primary: clean PB minus no-boost at the picked alpha, on test_open38, oov50, conv34. "Helps" = all
         3 seeds lower than their own no-boost WER and seed-mean gap >= 1 pt on that set, with test30 not worse by
         more than 0.5 pt. Item 1's closed biasing failed by repetition loops in Whisper's autoregressive decoder; PB
         works inside RNNT greedy decoding (one label per frame), so loops are not expected; count hyps with a word
         repeated 3+ times in a row anyway.
1. **DONE 2026-10-28, null (3.6).** **Contextual biasing / LM fusion for unseen words (decode-only, ~4 GPU-h).** The page says errors are
   very consistent across Whisper variants (so more ensembling has a ceiling) and lists "personalised phrase completion" as
   future work. Not in Tianyu's ablations. Apply to Tianyu-style best checkpoints we reproduce (encoder-only + old data):
   (a) shallow fusion with an n-gram LM over the patient's sentence material plus the 10k corpus, (b) Whisper prompt /
   keyword biasing with a caregiver word list, (c) N-best rescoring with a small LLM. OOV is the target; test30 must not
   get worse. Bar: semg1 OOV < 10.0, semg2 OOV < 2.1. 5 seeds of the base model, paired bootstrap.
2. **Audio-conditioned synthetic video for semg3 (~1-2 GPU-days).** Page: synthetic video "so far not successful as they
   need audio as conditions, which semg3 doesn't have". We have Higgs TTS: synthesize audio for new sentences (generic or
   semg1/semg2-adapted voice, or a voice matched to semg3's mouth timing), drive a lip-sync generator on semg3 face crops,
   add the clips to USR 2.0 training. Bar: semg3 OOV50 < 4.24. Gate first: generated mouth crops must be readable by the
   current semg3 lip model (its WER on synthetic clips < 30%) before any training.
3. **DONE 2026-10-28, primary passes (3.6).** **Stack the separately-winning semg1 ingredients, 5 seeds (~1 GPU-day).** encoder-only + old data + both patients +
   fine-tuned Higgs synthetic (13.0 single run) + weight averaging (12.4) + vote (10.3). It is not clear from the page whether
   the 10.0 headline already stacks all of them; the run confirms or improves it, with an interval. Prerequisite for item 1
   (we need that checkpoint family locally).
4. **TORGO at paper scale (old T2, ~1.5 GPU-days), lower priority.** Tianyu's TORGO is a single-run replication; our
   per-speaker Higgs adapters plus 5-15 h LibriSpeech text, 5 seeds. Note: Tianyu found TORGO doesn't help the patients,
   so this is only a TORGO benchmark result.

**New order (user 2026-10-29): after item 0 finishes, do items 5-8 next, then 2 and 4.**

5. **Synthetic-data study beyond Higgs: literature review, then benchmark.** First a literature review of TTS models
   that can be fine-tuned on a few minutes of dysarthric speech (voice-cloning / LoRA / adapter-capable; open weights).
   Then benchmark the top 2-3 against Higgs on the same pipeline (same filter, same Whisper E2 recipe, seeds 1-5). The
   factors to vary, one at a time:
   - TTS model: Higgs vs the shortlisted alternatives.
   - Fine-tuning strategy: zero-shot cloning, LoRA, full fine-tune, per-patient vs pooled adapter.
   - Generation material: what text we prompt with (patient 50-word set, open-vocab / caregiver phrases, LibriSpeech-
     style generic, conversational turns) and how much (dose curve).
   - Synthetic quality: intelligibility (ASR judge), speaker similarity, dysarthric-severity match. Check whether
     quality predicts the ASR gain.
   - Hard rule: no generated sentence may appear in any val, test, oov or s2ov/s3ov eval set, checked by exact and
     normalised string match before training.
   Gate: do the review and shortlist first; a benchmark run needs a model that passes the Higgs filter at a similar
   keep rate.
   - **Review done 2026-10-29: `docs/investigations/tts_synthetic_lit_review_2026-10-29.md`.**
     - Main finding: off-the-shelf zero-shot TTS (F5, Higgs) regresses dysarthric speech toward healthy, intelligible
       speech. The one published Higgs study (Singh et al., Interspeech 2026, TORGO) also finds that hybrid data does
       not beat real-only on average, which matches our null. The methods that help model rhythm and pathology
       explicitly (DARS, IndexTTS + pathology-prefix LoRA).
     - Shortlist: IndexTTS-2 + pathology LoRA, F5-TTS fine-tuned, VoxCPM2 LoRA, with Higgs as the reference.
     - Cheapest first: (step 0) check whether our filtered Higgs clips are easier than real clips for the ASR judge;
       then add an RIR post-processing arm on the existing Higgs clips. Neither needs a new generator.
   - **Step 0 done 2026-10-29 (`asr_aug/stack/diag_synth.py`, job 21326631; report at
     /scratch/pa2753/stack/data/diag_report.md).** Zero-shot whisper-large-v3 WER as a neutral judge, plus rate, pause
     fraction and WavLM-sv similarity to the real-train centroid. Kept pool-A vs real train:

     | patient | zs_wer mean | zs_wer capped at 1 | frac clips < 0.5 | sec/word | spk_sim |
     |---|---|---|---|---|---|
     | s1 real train / synth A kept | 1.059 / 0.933 | 0.909 / 0.924 | 0.04 / 0.02 | 0.865 / 0.870 | 0.921 (LOO) / 0.890 |
     | s2 real train / synth A kept | 0.302 / 0.462 | 0.269 / 0.418 | 0.73 / 0.57 | 0.883 / 0.834 | 0.951 (LOO) / 0.954 |

     - Pre-registered rule: s1 formally "easier" (gap +12.6 pp); s2 not (synthetic 16 pp *harder*).
     - The s1 gap is an artefact: Whisper is at the floor on s1 (median WER 1.0 in every group), and the gap comes
       entirely from hallucinated insertions on the real clips. Capped at 1, synthetic and real are equal (0.924 vs
       0.909).
     - **Reading: the cleanup failure (lit review 1.1) does not explain our Higgs null.** Kept clips are as hard as
       real (s1) or harder (s2), and speaking rate and speaker similarity are close to real. The s2 extra difficulty is
       more likely TTS artefacts than pathology. The only clear acoustic gap is pause/silence fraction (synthetic about 2x,
       0.011-0.016 vs 0.006-0.007), which fits the RIR/channel-mismatch idea.
     - Selecting for "harder" synthetic clips is therefore not supported. Next cheapest step stays the RIR + room-noise
       arm on the existing clips. 12 clips under 1 s were zero-padded for spk_sim; all are in rejected groups.
   - **RIR arm E2R pre-registered 2026-10-29 (user go-ahead), before any RIR clip existed:** design and rule in
     `asr_aug/stack/README.md` ("E2R"). E2 with the same draw and recipe, synthetic clips room-simulated (simulated RIR +
     the patients' own noise floor at real-matched SNR). Primary E2R - E2 on s1_oov, 5 seeds, helps = CI upper < 0.
   - **E2R result 2026-10-29: null** (`asr_aug/stack/results/report_rir.out`; seeds 1-5, val-selected checkpoint).
     Primary E2R - E2 on s1_oov: **+0.81 pp [-3.30, +5.27], p 0.72**, so the rule (CI upper < 0) is not met. WER % E2 vs E2R
     (single-seed mean): s1_test 7.57 vs 6.27 (-1.30 [-4.07, +1.00]), s1_oov 15.16 vs 15.96, s2_oov 2.63 vs 2.46,
     s2_test 0.59 both. Soup: s1_oov 8.07 vs 8.52, s2_oov 1.75 vs 1.32. Last epoch agrees (s1_oov +0.72 [-2.31, +3.69]).
     Room/channel mismatch is not what limits the Higgs clips either. Together with step 0, the cheap no-new-generator
     fixes are exhausted. The remaining item-5 route is a different generator (shortlist: IndexTTS-2 + pathology LoRA,
     F5-TTS fine-tuned, VoxCPM2 LoRA).
   - **Generator benchmark drafted 2026-10-30: `asr_aug/gen/README.md`** (not started; needs the user's go and an OK
     on the bilibili licence). G1 = IndexTTS-2 + per-patient speaker LoRA; G2 = ProtoDisent (IndexTTS-1.5 +
     pathology prefix + GRL), trained pooled with the TORGO controls (not on the cluster yet). Gate on keep rate and
     closeness to real (step 0 metrics) vs Higgs, then E2Gx - E2 on s1_oov, 5 seeds, Bonferroni over generators.
   - **G1 gate 2026-11-01: FAIL on s1 and s2, so no Step B for G1** (`results/gen_gate/g1_gate_report.md`). G1 is
     only closer to real than Higgs on spk_sim. Its speech is about 55% too slow: 1.35 sec/word against 0.87 real
     (s1) and 0.88 (s2), where Higgs is 0.92 / 0.89. Most rejections are duration rejections. Keep rate is 0.23 vs
     Higgs 0.38 on s1, and 0.36 vs 0.88 on s2.
   - **G1d (G1 + s2mel duration control, README amendment 2) 2026-11-02: FAIL on s1 and s2**
     (`results/gen_gate/g1d_gate_report.md`).
     - The rate is fixed: 0.81 / 0.85 sec/word, and keep rate is at least 0.5x Higgs for every voice.
     - The 2x compression costs s1 its voice match (spk_sim 0.76, against Higgs 0.89) and s2 its intelligibility
       (zs_wer 0.68, against real 0.27).
     - The G1 line is closed. G2 (ProtoDisent) was run next (README amendment 3).
   - **G2 (ProtoDisent, README amendment 3) 2026-11-03: FAIL on s1 and s2** (`results/gen_gate/g2_gate_report.md`).
     - Best keep rates of any generator (s1 0.70 vs Higgs 0.38), but closer to real than Higgs on only one metric
       per patient: s1 on rate, s2 on zs_wer.
     - s1 is far too intelligible (zs_wer 0.52 vs real 0.91), the voice match is weak (spk_sim 0.68 s1, 0.82 s2,
       against Higgs 0.89 / 0.95), and clips have many more pauses than real speech.
     - **Item 5's generator benchmark ends here, as pre-registered: no generator passed the gate, so no Step B (no
       E2Gx ASR arms).** Higgs stays the synthetic generator.
6. **Diffusion- and GAN-based audio generation (user idea 2026-09-28).** Look beyond Higgs's LLM-style (discrete-token)
   generation: diffusion or flow-matching TTS and GAN vocoders, or voice conversion of healthy speech into the patient's
   voice. Either use one as a generator in item 5's benchmark, or bring a useful technique into Higgs (e.g. a
   diffusion/flow decoder or a better vocoder). Covered by the same literature review as item 5.
7. **Conversation-style data (user idea 2026-10-09, followed up 2026-10-13).** Adapt the collection approach of the
   artificial-conversations ASR paper (to be discussed with Adeen). Collect real conversation recordings to test how the
   models do on actual dialogue, which none of our current test sets cover. Tianyu's related idea: TV-show dialogues and
   conversation history as a decoding prior. Conversational text is also a candidate generation material for item 5.
8. **Test-time adaptation (arXiv 2610.23117, Wang et al., NCMMSC 2026).** That paper builds a sentence-level dysarthric
   AVSR corpus from CDSD videos. It then applies unsupervised test-time adaptation with SGEM (sequence-level generalised
   entropy minimisation) and reports significant CER gains on unseen dysarthric speakers. For us: try SGEM on our
   fine-tuned Whisper / Nemotron for a new session or a new patient (semg1 new data, s2ov). It is decode-time only,
   no labels needed.

### 3.4 Closed (ours)
H2 target-vocab synthetic (null); semg2 per-patient vs pooled (floor); earlier list in 1.2.
Contextual biasing as pre-registered (prompt / token 4-gram fusion / Qwen2.5-1.5B rescoring, no loop guards): null,
repetition loops (3.6). Stack E2 vs E1: done, helps OOV (3.6).

### 3.5 Shared-split eval set up (2026-10-26)
Splits are Tianyu's own tree (th3482 = Tianyu He): our s1/s2 labels are `LATEST_combined/roles/s{1,2}_*` (now combined_v6), so the rows are already shared.
Built `asr_aug/shared_eval/score_shared.py` (refs from the role file; aborts on row mismatch); validated: N s1_test 6.98 = report 7.0. Our table in `asr_aug/shared_eval/README.md`.
Side results: multi-patient (section 2) runs finished. semg2 is near the floor (large-v3 FT s2_test 0.6-1.2%, oov 5.7-6.0%); pooled s1+s2 no different on s2; the prompt choice doesn't matter on s2.
Large-v3 zero-shot s1_test 85% vs Tianyu's medium zero-shot 0.631 on "Patient", so Tianyu's patient set probably is not s1_test.

### 3.6 Results: item 3 (stack, 5 seeds) and item 1 (contextual biasing), 2026-10-28
Design and amendment: `asr_aug/stack/README.md`. Full outputs: `asr_aug/stack/results/report_{stack,bias}.out`.
Large-v3 encoder-only, seeds 1-5. E1 = old split_v1 + s1_train. E2 = E1 + s2_train + 397 Higgs clips (semg1 old-adapter 138, Oct 1 adapter 139, semg2 adapter 120).

**Item 3: stacking helps semg1 OOV (primary passes).** Val-selected checkpoint, WER % test / OOV:
| system | s1 (semg1) | s2 (semg2) |
|---|---|---|
| E1 single (mean ± sd) | 5.09 ± 2.24 / 21.17 ± 2.14 | 11.60 / 18.33 (not trained on s2) |
| E2 single | 7.57 ± 1.35 / 15.16 ± 2.23 | 0.59 / 2.63 |
| E2 soup (5-seed weight average) | 7.10 / **8.07** | 0.59 / **1.75** |
| E2 MBR vote | 7.10 / 9.87 | 0.59 / 1.75 |
| Tianyu best | 4.1 / 10.0 | 0.6 / 2.7 (Nemotron 3.0 / 2.1) |
- Primary E2 - E1, s1_oov, single seeds: **-6.01 pp [-11.62, -0.98], p 0.020**. Last epoch agrees: -6.46 [-11.24, -2.07].
- s1_test: E2 - E1 +2.49 [-0.81, +6.86], so no test gain (test is near its floor, 30 sentences).
- Soup beats single seeds on s1_oov in both arms: E2 -7.09 [-11.43, -3.08], E1 -5.02 [-10.28, -0.60]. MBR also helps on E2 (-5.29) but less than the soup.
- E2 soup beats Tianyu's bar on semg1 OOV (8.07 vs 10.0) and semg2 OOV (1.75 vs 2.1/2.7). It is still behind on semg1 test (7.10 vs 4.1). Descriptive only: one soup model, 49 OOV sentences; Tianyu's numbers are single runs.
- E2 changes two things (semg2 real + synthetic), so the gain cannot be split between them.
- Synthetic filter caveat: the s1 judge kept only 31%/45% of the new semg1 clips and probably favours easy sentences.

**Item 1: no biasing method helps (all primaries fail).** Clean method - beam, s1_oov, single seeds, Holm over 3:
| arm | prompt | n-gram fusion | LLM rescoring |
|---|---|---|---|
| E1 | +35.96 [+0.45, +106.95], Holm 0.11 | +2.51 [-1.35, +6.55], 0.37 | +0.36 [-2.97, +3.55], 0.83 |
| E2 | +11.57 [-3.41, +68.79], 0.96 | +25.29 [-0.42, +102.91], 0.21 | +11.48 [-3.98, +67.31], 0.96 |
- On the best system (E2 soup) every method is no better than plain beam on s1_oov (beam 8.07; prompt 9.42, LLM 8.52 / words 8.07). The words condition (OOV words given) does not help either.
- The main failure is repetition loops, which push WER past 100%:
  - Prompt biasing with the full word list (words condition) loops often: 17 loops on E1 s1_oov over 5 seeds.
  - n-gram fusion loops grow with lam (E1 s1_oov 0/0/6/15 loops at lam 0.1/0.2/0.3/0.5). On E2 it loops even at lam 0.1 (E2 s2_oov: 8 loops, mean WER 89.9 vs beam 2.7). E2_seed5's val picked lam 0.3.
  - LLM rescoring is near-neutral at mu <= 0.025. Higher mu picks looped 10-best candidates (E2 s2_oov 2.7 at mu 0.025, 52 at mu 0.1). The mean token log-prob rewards repetition.
- Amendment 1 (finer mu grid) changed LLM from clearly harmful to neutral; it did not make it help.
- Possible follow-up (not pre-registered): loop guards (no-repeat n-gram, EOS-aware fusion, a length or repetition penalty on the LLM score). The gain available is small: E2 soup beam is already 8.07 on s1_oov.

### H2 results (2026-10-28; 20/20 jobs done, `asr_aug/report_h2.py`, seeds 1-5, oov rows of the held-out fold only)
Val-selected ckpt oov WER %: N 31.0, NF 25.1, NS 26.5, ON 27.4, ONF 26.4, ONS 21.4.
Primary (Holm over 2): NF-N -5.9pp [-12.7,+0.9] Holm p=0.17; ONF-ON -1.1pp [-7.0,+5.2] Holm p=0.72. Neither "helps" (CI upper not < 0).
Secondary: NF-NS -1.4 (ns); ONF-ONS +4.9 [-0.6,+11.1] (targeted text not better than generic; trends worse). Last ckpt: NF-N -0.2, ONF-ON +0.5, ONF-ONS +4.8.
Verdict: oov-sentence text from the other fold does not help beyond noise; generic synthetic (ONS) is the best arm.
