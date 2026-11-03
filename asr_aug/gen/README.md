# Item 5 generator benchmark: IndexTTS first (PLAN.md 3.3 item 5)

Draft pre-registration, 2026-10-30. No IndexTTS model, clip or ASR run exists yet. This becomes binding when the user
gives the go-ahead; any change after that is logged as a dated amendment below.
Cluster mirror (planned): `/scratch/pa2753/gen/{code,models,data,root,runs,logs,jobs.txt}`.

## Why this, why now
The Higgs route is closed. Synthetic E2 did not beat real-only, step 0 showed Higgs clips are not "too clean", and the
RIR arm E2R was null (PLAN item 5). What is left is a generator that models pathology apart from timbre. The lit review
(`docs/investigations/tts_synthetic_lit_review_2026-10-29.md`) found one method with released code that does this:
ProtoDisent (Wang et al., arXiv 2602.08696), built on IndexTTS. We test two IndexTTS variants, with Higgs as the
reference. F5-TTS and VoxCPM2 come later, and only if a variant here passes the gate.

## Generators
- **G1 = IndexTTS-2 + speaker LoRA, per patient.** Built on [risingore/index-tts-2-lora](https://github.com/risingore/index-tts-2-lora)
  (MIT): LoRA r16 on all 24 GPT layers, a learnable speaker condition, 15 epochs, lr 5e-5.
  - It is trained on that patient's train role only, like the Higgs adapters in `asr_aug/stack`: s1 Oct 1 train (240)
    plus old split_v1 train (314), and s2 train (240).
  - It is the direct swap for Higgs: same per-patient adapter setup, different codec LM.
- **G2 = ProtoDisent (IndexTTS-1.5 + pathology prefix + GRL + VC counterfactuals), pooled.** Built on
  [Mors20/Controllable-Dysarthric-Speech-Synthesis](https://github.com/Mors20/Controllable-Dysarthric-Speech-Synthesis),
  the paper's official code. No trained checkpoints are released, so we train it.
  - Pathology IDs: 0 = healthy, 1 = semg1, 2 = semg2. TORGO's 8 dysarthric speakers are added as extra IDs, as in the
    paper, to give the disentangler more pathologies. TORGO rows are used for generator training only, never for ASR.
  - Healthy speakers: the TORGO controls (FC01-03, MC01-04). They are **not on the cluster** (`/scratch/th3482/TORGO/raw`
    has only the 8 dysarthric speakers), so they need a download into `/scratch/pa2753/gen/data`.
    Fallback: a LibriTTS-R subset of matched size.
  - Generation uses the patient's own timbre prompt plus that patient's pathology prefix. The pathology-on-healthy-voice
    mode is a later factor, not part of this study.

## Setup checks (A0, before any training; each one is a stop condition)
1. **Code really implements GRL and pathology prefixes.** Read `train.py` and `indextts/` in the Mors20 repo; the
   README alone does not show it. If it is missing, G2 is dropped and logged here.
2. **Environment.** New env `indextts` (Python 3.11, torch 2.8 cu128, which has sm_100 kernels). A B200 smoke test
   generates one clip from the stock model. G1 and G2 pin different stacks (transformers 4.46.3 / peft 0.20 for G2),
   so they may need two envs.
3. **Licence.** Both IndexTTS-1.5/2 are under the bilibili Model Use License. Clause 3.4(c) forbids using outputs to
   improve other AI models "except for... non-commercial AI models"; our ASR is non-commercial research. Clause 4.2
   bars deployment "in high-risk scenarios such as medical diagnosis"; ASR augmentation is not diagnosis.
   **This needs the user's (and, if needed, the lab's) OK before step A1.**
4. **Leakage guard (hard rule).** Same text plan as Higgs: `stack/data/plan_gen.json`, pool A/B text minus every s1/s2
   val/test/oov and s2ov/s3ov sentence after normalisation. `check_leak.py` re-checks every generated row and fails the
   job before any ASR training.

## Step A: generator gate (no ASR training)
Per generator and patient, generate the **full** pool plan (the same rows Higgs got), then:
- **Filter:** the frozen H filter from `asr_aug/stack/filter_synth.py`, with the same judges and thresholds.
- **Gate metrics:** the step 0 diagnostics (`diag_synth.py`), computed on the kept clips:
  zero-shot whisper-large-v3 judge WER capped at 1, sec/word, pause fraction, and WavLM-sv similarity to the
  real-train centroid.
- **Gate rule, per patient.**
  - Keep rate must be at least 0.5x Higgs's keep rate. This is PLAN item 5's "passes the Higgs filter at a similar keep
    rate".
  - The kept clips must be at least as close to real as Higgs-kept on 2 of the 3 measures |capped judge WER - real|,
    |sec/word - real| and spk_sim.
  - A generator goes to Step B only if it passes the gate on s1, the primary patient.
- Listening check: 20 random kept clips per generator and patient, written to `results/gen_listening/` for the user.
  Descriptive only; it does not decide anything.

## Step B: ASR arms (only for generators that pass the gate)
- **E2G1 / E2G2** = E2 from `asr_aug/stack`, with each Higgs clip replaced by that generator's kept clip for the same
  text row.
  - Same draw: rng key `stack-E2-{seed}`, as E2R.
  - Same count per voice. A text row the generator's filter rejected is replaced by the next kept row of the same
    pool, in fixed order.
  - Same recipe: `whisper_stack.sbatch`, large-v3 encoder-only, seeds 1-5, val-selected checkpoint.
- **Primary:** E2Gx - E2 on s1_oov.
  - Paired two-level bootstrap from `report_stack.py`, 4000 draws.
  - "Gx beats Higgs" = upper bound of the 97.5% CI < 0. That is Bonferroni over 2 generators; with one generator in
    Step B, the 95% CI is used.
- **Secondary (CIs, no claims):**
  - E2Gx - E1 (does synthetic help at all over real only).
  - s1_test, s2_test, s2_oov and s2ov_test.
  - Soup vs soup.
  - The last-epoch checkpoint.
  - Spearman correlation between the gate metrics and the per-arm s1_oov WER. This answers "does quality predict gain"
    across Higgs, E2R, G1 and G2; with 3-4 points it is descriptive only.

## Compute (preemptible, resume-safe)
- **G1:** extract plus ~15 epochs is under 1 GPU-h per patient. Generating ~4k clips is about 2 GPU-h. B200.
- **G2:** Seed-VC counterfactuals, then pooled training. The repo states no time; budget ~1 B200-day. Generation is
  ~2 GPU-h.
- **Step B:** 10 Whisper large-v3 runs on l40s/a100/h100/h200. The whisper_asr env has no B200 kernels.
- Resume: TTS training writes checkpoints at least every 30 min; a G1/G2 run without resume is not submitted to a
  preemptible partition.
  Whisper reuses `--resume`.

## Known limits
- 49 s1 OOV sentences: effects under ~4 pp are unlikely to resolve with 5 seeds (as in E2R).
- G2 changes two things against Higgs at once: the base model and the pathology conditioning. G1 separates the
  base-model effect.
- The paper's gains are on TORGO under leave-one-speaker-out, where real data for the test speaker is absent. Our setup
  has the speaker's real train data, which is the setting where the Higgs study (Singh et al.) found no average gain.

## Order
1. A0 checks.
2. G1 gate.
3. G2 gate. It can run in parallel with G1 Step B once A0 passes.
4. Step B for the generators that passed.
5. Report: `report_gen.py`, results in PLAN item 5.

## A0 log
- 2026-10-30: user approved the plan and the licence reading ("Approve, start A0").
- Check 1 passed. Mors20 `train.py` (commit 6e594f8) has `GradientReversal`, with lambda ramped to 0.1 over 500 steps.
  It also has learnable `mean_pathology_condition_k` prefixes, two binary pathology classifiers (one on the summed
  condition, one on the speaker condition behind the GRL) and LoRA+. Notes on the repo:
  - Its convention puts the mild TORGO speakers (F03/F04/M03) in the healthy class 0.
  - Its Seed-VC counterfactual step pairs speakers by identical TORGO prompt slots. Our patients share no prompts with
    TORGO, so G2 needs an adaptation there (to be specified before G2 starts).
- Check 2: two envs, set up by `setup_a0.sh`.
  - `idx2`: index-tts d9e41aa, torch 2.8, Python 3.11, peft 0.17.
  - `pdis`: Mors20 with Python 3.10.
  - pdis needed two fixes. The repo pins conflict, so `tokenizers` is overridden to 0.20.3 while keeping
    `transformers==4.46.3`. `webrtcvad` is swapped for the prebuilt `webrtcvad-wheels`, because the login node has no C
    compiler.
  - B200 smoke test passed: job 21444084, gb003, `smoke.sbatch`, clips in `results/gen_smoke/`.
    - Both envs run on sm_100 (torch 2.8.0+cu128).
    - Stock IndexTTS-2 spoke a 12-word sentence in 10.7 s (s1 voice) and 8.0 s (s2), loading in 55 s with peak
      memory 7.9 GB.
    - IndexTTS-1.5 took 7.6 s (s1) and 15.8 s (s2).
    - pdis loaded 1.5's gpt.pth into ProtoDisent's UnifiedVoice: 583 M params.
    - Note: the ft_stitch5 references are 25-26 s, and IndexTTS uses all of it as the prompt.
- Weights are in `gen/models/{IndexTTS-2, IndexTTS-1.5}`, plus the IndexTTS-2 aux models (w2v-bert-2.0, campplus,
  MaskGCT semantic codec, BigVGAN).
- TORGO controls FC01-03 and MC01-04 were downloaded to `gen/data/torgo_dl`. `gen/data/TORGO` holds all 15 speakers:
  the dysarthric ones are symlinks to th3482's read-only copy.
- Check 4 passed (`check_leak.py`). The generation text `stack/data/text.json` (1,998 sentences) has 0 matches among the
  347 held-out sentences from all 29 non-train roles, including s2ov, s3ov and old_*.
  The guard used for Higgs (`make_gen_plan.py`) checked only s1/s2 val/test/oov.

## Amendments
1. **2026-10-30, user decision "Match Higgs (A)" (before any G1 training): G1 is one IndexTTS-2 LoRA per Higgs adapter.**
   This replaces the single pooled semg1 adapter in the draft. Each adapter is trained on exactly the same `tts_train`
   clips as its Higgs counterpart and regenerates that adapter's (text, reference) rows (`prep_g1.py`).
   - **old:** `higgs_ft/data/splits.json`, 283 clips. This is the ONS adapter whose pool fills half of E2's semg1
     synthetic share. Plan: `asr_aug/data/plan_gen.json`, 4 refs x 500 rows.
   - **s1:** `stack/higgs/s1`, 216 clips, 1,998 rows.
   - **s2:** `stack/higgs/s2`, 216 clips, 1,998 rows.

   Training uses the risingore defaults (r16, lr 5e-5, LoRA+ x8, batch 4, seed 1). The **final epoch 15 is fixed in
   advance**, as Higgs's epoch 8 was; there is no selection on validation. At generation the voice comes from the
   same reference wav Higgs used. The learned speaker vector is a training aid only, because the tool's inference
   conditions on the reference.

   The ONS plan has 2 rows whose text is now s1/s2 oov ("I feel hot now", "The computer is not working"). They are
   dropped (1,998 rows remain). The audit of where they went:
   - E1/E2/E2R never contained them: `make_mix.py` filters them out.
   - They do appear in H2 `oct1_*F_foldB` training. That is by design: fold B trains on them and is scored only on
     fold A sentences.
   - They also appear in the old split_v1 whisper/nemotron M/S runs, which were scored on the split_v1 test set. That
     set does not contain them.
2. **2026-11-01, user decision "G1 + duration control", after the G1 gate failed: new arm G1d.**
   - **What G1d is.** The same three merged G1 models, regenerating the same plan rows (same seeds and references), with
     one change: the speaking rate is scaled back to the real rate.
   - **Why.** G1's clips are 97-99% speech (median speech fraction), so the slowness is in the articulation, not in silence
     padding. Their median sec/word is 1.47 (old), 1.60 (s1) and 1.78 (s2), against real 0.86 / 0.845 / 0.861. Stock
     IndexTTS-2 in the smoke test was near the real rate, so the LoRA caused the slowdown.
   - **How.** The released IndexTTS-2 code has no token-count duration control in the GPT: `speed_emb` is always fixed at
     indices 0 and 1. Duration is therefore set at the s2mel stage, where `infer_v2.py` computes the mel length as
     `code_lens * 1.72`. That product is multiplied by a fixed per-voice factor `dur_scale`, a one-line patch applied by
     `g1d.sbatch`; the default of 1.0 reproduces G1. The flow-matching model then renders the same semantic codes over
     fewer frames, which changes the rate without changing the pitch.
   - **The factor.** Real-train median sec/word divided by G1's median sec/word over all of its generated rows:
     **old 0.586, s1 0.528, s2 0.483**. It is fixed now and never tuned on the gate or on any eval set.
   - **Gate.** The rule is unchanged (Step A). Caveat: the rate measure is calibrated toward real by construction, so G1d
     passing on "rate + spk_sim" says less than a pass for G1 would have. Step B remains the real test.
   - **Correction for multiple tests.** If G1d and G2 both reach Step B, the 97.5% CI applies; if only one does, 95%.
   - **Smoke before the full run.** 20 rows per voice. It is a stop condition only if a clip is clearly broken (silent or
     garbled); no settings are tuned on it.
3. **2026-11-02, G2 specification and counterfactual adaptation. Approved by the user ("Approve, start G2"), with two
   corrections made before anything ran (see "Balancing" below).**
   Repo: Mors20 6e594f8. Its pipeline is Seed-VC counterfactuals (`third_party/seed-vc/generate_data.py`), then
   features (`scripts/prepare_torgo.py`), then `train.py`.
   - **What the repo does.** Seed-VC converts timbre only: the *source* utterance supplies content, prosody and
     articulation, and the *reference* supplies timbre. The repo builds two sets:
     - `X_converted_from_Y`: dysarthric content from Y in the healthy/mild timbre of X, labelled class(Y). There are
       33,795 such pairs on our TORGO copy.
     - `X_converted`: healthy content in patient X's timbre, labelled 0. There are 2,990 of these.
     - Pairs are matched on identical TORGO prompt text. That matching is only bookkeeping: the output takes the text
       of the slot it fills, which equals the source's text. Seed-VC itself does not need the texts to match.
   - **Adaptation for our patients.** The text comes from the content source, so no prompt overlap with TORGO is
     needed.
     - Patient clips are the same `tts_train` clips the Higgs adapters used: semg1 = old (283) + s1 (216) = 499 clips,
       semg2 = s2 (216 clips).
     - **(a) Pathology content in healthy timbres, labelled with the patient's class.** Every patient clip is the
       source. It is converted into each of the repo's 10 timbre targets (FC01-03, MC01-04, F03, F04, M03), with a
       reference drawn seeded from that speaker's utterances of 3 or more words. That gives 715 x 10 = 7,150 clips.
       Text = the patient clip's transcript.
     - **(b) Healthy content in the patient's timbre, labelled 0.** Every patient clip is the timbre reference. The
       source is a seeded random TORGO control utterance of 3 or more words (2,921 available, 322 unique texts). That
       gives 715 clips. Text = the control utterance's prompt.
     - The repo's own TORGO sets (both modes) are kept unchanged, so the TORGO pathologies are trained as in the paper.
     - VC settings are the repo's: 30 diffusion steps, cfg 0.7, no f0, fp16, seed 1234.
   - **Classes.** The repo's mapping is kept: 0 = healthy, which includes F03/F04/M03 and every `X_converted` clip;
     1-5 = F01, M01, M02, M04, M05. Two classes are added: **6 = semg1** (old and s1 together: same patient) and
     **7 = semg2**. `num_classes` becomes 8. Each prefix starts as the mean condition of its class, as the repo does.
     This replaces the draft's numbering of 1 = semg1 and 2 = semg2.
   - **Training length and sampling: the repo as released (corrected before any run, user decision).**
     - The draft assumed a cap of 150 batches per epoch and so planned a class-balanced sampler. Both were wrong.
     - `train.py` never reads `max_steps_per_epoch`, `gradient_accumulation_steps` or `early_stopping_patience`. The
       README command trains full epochs: about 54k items at batch 2, i.e. about 27k steps per epoch, for 20 epochs.
     - The patients make up about 15% of items once their counterfactuals are counted (semg1 about 5.5k, semg2 about
       2.4k), not 1.3%. A TORGO patient class has about 7.5k items and healthy about 17k.
     - User decision: train as released, i.e. `--epochs 20 --batch-size 2`, uniform shuffle, no balancing. The
       remaining settings are the repo's: LoRA+ x8, lr 5e-5, GRL ramp to 0.1 over 500 steps, seed 91.
   - **Checkpoint.** The repo's `gpt_best.pth`, chosen by val mel loss on its own per-speaker 2% val split. That split is
     internal to the generator and holds no ASR eval data.
   - **Resume.** `train.py` has no resume. An epoch-level resume will be added (unmerged LoRA, pathology params,
     optimizer, scheduler, RNG and best-loss state) before any preemptible submission.
   - **Generation.** IndexTTS-1.5 + G2 GPT (`indextts.inference --pathology k`). The rows and references are the same
     as Higgs and G1: old and s1 use class 6, s2 uses class 7. That is 3 x 1,998 clips. The seed is md5(rel). There
     is no duration or rate post-processing.
   - **Leak guard.** `check_leak.py` runs on the generation plan (already clean) and on every text in the patient
     counterfactual manifests, i.e. the patient tts_train transcripts and the chosen TORGO control prompts.
     - Result (2026-11-02): 45 of the 283 old-voice tts_train clips have text that is in `old_legal_val` (30) or
       `old_legacy_val` (15). The same real clips trained the Higgs old adapter and G1-old. Their 450 type-(a)
       counterfactual pairs are dropped (`g2_cf.py`), so those sentences are never synthesised, while the 45 real clips
       stay in training as they did for Higgs and G1. That leaves 7,415 patient pairs (semg1 4,540 (a) + 499 (b),
       semg2 2,160 (a) + 216 (b)), and the recheck finds 0 leaks.
   - **Compute.**
     - Weights to download: Seed-VC `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned`, funasr/campplus and the
       BigVGAN 22k vocoder. whisper-small is already cached.
     - Conversions: about 44.6k, roughly 12 B200 GPU-h. Feature preparation: a few GPU-h.
       Generation: about 2 GPU-h. Training at full epochs is about 1 B200-day (to be measured in the smoke run).
     - Code: `g2_cf.py` / `g2_cf.sbatch` (counterfactuals), `g2_prep.py` / `g2_prep.sbatch` (features),
       `g2_train.py` / `g2_train.sbatch` (the repo's Trainer, with an exact mid-epoch resume added), and
       `gen_g2.py` / `g2_gen.sbatch`. The gate reuses `gate_g1.py` with `GEN=g2`.
     - Every batch-1 stage runs NW=4 sharded workers per GPU and logs nvidia-smi. HPC cancelled job 21613869 for low
       GPU utilisation.
   - **Gate and stop rule.**
     - The Step A rule is unchanged, and s1 decides.
     - G1 and G1d are out, so if G2 reaches Step B the 95% CI applies.
     - **If G2 fails the gate, the item 5 generator benchmark ends.** There is no post-hoc rescue arm (the lesson of
       G1d), and F5-TTS and VoxCPM2 stay unrun, as the "Why" section requires.
   - **Licences.** IndexTTS-1.5 is under the bilibili licence already approved. Seed-VC is GPL-3.0: we only run it and
     do not redistribute it.

## Results
- **2026-11-01, G1 gate: s1 FAIL, s2 FAIL, so G1 does not go to Step B.** Job 21610585 (`gate_g1.sbatch`, 39 min);
  report in `results/gen_gate/g1_gate_report.md`.
  - Keep rate (G1 vs Higgs): old 0.50 vs 0.77, s1 0.23 vs 0.38 (passes the 0.5x rule), s2 0.36 vs 0.88 (fails).
  - On the kept clips, G1 is closer to real than Higgs only on spk_sim (s1 0.913 vs 0.890, s2 0.972 vs 0.953).
    - sec/word: G1 1.35 / 1.36 against real 0.87 / 0.88 (Higgs 0.92 / 0.89).
    - Capped zs_wer: s1 G1 0.825 vs real 0.909 (Higgs 0.916); s2 G1 0.468 vs real 0.269 (Higgs 0.398).
  - Rejections, given as dur / dur+judge / judge-only out of 1,998:
    - s1: 171 / 708 / 665.
    - s2: 613 / 471 / 205.
    - old: 173 / 468 / 354.
  - Reading: the IndexTTS-2 LoRA copies the voice better than Higgs, but its speech is far too slow.
- **2026-11-02, G1d gate (amendment 2): s1 FAIL, s2 FAIL, so G1d does not go to Step B.** Job 21634962 (36 min); report
  in `results/gen_gate/g1d_gate_report.md`. Generation ran as 21613869/71/73, with the old voice finished by 21630707
  after HPC cancelled 21613869 for low GPU utilization.
  - Duration control fixed the rate.
    - sec/word: s1 0.81 vs real 0.865, s2 0.85 vs real 0.883.
    - Duration rejections fell to 6-22 per voice.
    - Keep rates: old 0.565, s1 0.237, s2 0.484, all >= 0.5x Higgs.
  - What the speed-up cost:
    - Judge rejections rose: s1 1,408, s2 975 and old 752 judge-only out of 1,998.
    - s1 spk_sim fell to 0.764, from 0.913 in G1 (Higgs 0.890).
    - s2 capped zs_wer is 0.681, against 0.269 real and 0.398 for Higgs.
  - Closer than Higgs: s1 only on rate; s2 only on spk_sim (0.957 vs 0.953).
  - Reading: compressing the semantic codes about 2x at the s2mel stage removes the slowness, but it costs either the
    voice match (s1) or intelligibility (s2). The slow articulation comes from the LoRA GPT itself, and a fix at
    rendering time does not recover it. **The G1 line (IndexTTS-2 + per-voice LoRA) is closed.**
- **2026-11-03, G2 gate (amendment 3): s1 FAIL, s2 FAIL, so G2 does not go to Step B. Per the pre-registration this
  ends the item 5 generator benchmark: no generator passed, and Step B is not run.** Gate job 21739437 (32 min,
  pr_60_general race winner); report in `results/gen_gate/g2_gate_report.md`.
  - Pipeline: counterfactuals 21661574-77 (43,859 converted, 341 failed, <1%); prep 21661578 (94 dirs, 60,918 items,
    all 8 classes filled); training 21661580 (20 epochs, 12h08m, one preemption resumed at epoch 2 batch 24,000;
    best val mel loss 3.539 at epoch 17, flat after); generation 21736128-30 (5,994 clips, ~3 s/clip per worker).
  - One code fix before generation: `gen_g2.py` first loaded the GPT in fp16, which gave NaN sampling probabilities
    (first generation jobs 21719954-56 died on a multinomial assert). It now runs fp32, as the repo's own inference does.
  - Keep rate is the best of any generator: old 0.798, s1 0.702, s2 0.969 (Higgs 0.770 / 0.383 / 0.876). Rejections
    are almost all judge rejections (s1 594, old 404, s2 61 out of 1,998); duration rejections are 0-2 per voice.
  - On the kept clips, G2 is closer to real than Higgs on one metric per patient, never two:
    - s1: rate only. sec/word 0.875 vs real 0.865 (Higgs 0.916). zs_wer 0.524 vs real 0.909 (Higgs 0.916); spk_sim
      0.680 (Higgs 0.890).
    - s2: zs_wer only. 0.199 vs real 0.269 (Higgs 0.398); sec/word 0.809 vs real 0.883 (Higgs 0.891); spk_sim 0.822
      (Higgs 0.953).
  - Pause fraction is far above real: s1 0.138 vs 0.007, s2 0.058 vs 0.006 (Higgs 0.016 / 0.011). Part of the s1
    rate match is silence, not slower articulation.
  - Reading: the pathology prefix with GRL gives clean, intelligible speech in roughly the right voice family, but it
    does not carry s1's severity (zs_wer 0.52 against 0.91 real) and loses the per-speaker voice match that
    Higgs's speaker adapter has (spk_sim 0.68 / 0.82). Pooled class prototypes trade speaker identity for control.
