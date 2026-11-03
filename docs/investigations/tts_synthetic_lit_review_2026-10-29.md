# Synthetic dysarthric speech beyond Higgs: literature review (PLAN.md 3.3 items 5 and 6)

Search date 2026-10-29. This is a scoping review, not a systematic one. Sources: web search (four broad queries and four
targeted ones) plus arXiv abstract pages, read for every paper cited below. A claim marked *(secondary)* comes from a
blog or listing and was not checked on the primary source. Licences especially must be checked on the model card
before use.

## 1. What the literature says, by theme

### 1.1 Most off-the-shelf TTS "cleans up" dysarthric speech
- **F5-TTS zero-shot cloning on TORGO** "exhibits a strong bias toward speech intelligibility over speaker and prosody
  preservation", and the bias varies with severity ([Anuprabha et al., Interspeech 2025, arXiv 2508.05102](https://arxiv.org/abs/2508.05102)).
- **RTFree-F5** says F5-TTS's text-based reference conditioning carries atypical patterns into the output. The authors
  treat that as a defect and fix it: synthesis WER on dysarthric references falls from 24.6% to 10.4%
  ([Eom et al., Interspeech 2026, arXiv 2606.20266](https://arxiv.org/abs/2606.20266)). For augmentation we want the
  **opposite**, so any model tuned for intelligibility is a poor fit as it comes.
- **Implication for us:** the property to select for is *pathology preservation* (synthetic clips should be as hard
  for an ASR judge as the patient's real clips), not naturalness or intelligibility. Our current Higgs filter keeps
  clips the judge can read, which may push the set toward easy, cleaned-up speech. This is worth checking on our own
  data first (section 3, step 0).

### 1.2 Higgs zero-shot cloning: the closest published replication of our setup
- [Singh et al., Interspeech 2026, arXiv 2606.19823](https://arxiv.org/abs/2606.19823) clone TORGO speakers with **Higgs
  Audio V2** (zero-shot) and fine-tune Whisper-medium:

  | condition | WER |
  |---|---|
  | zero-shot | 31.62% |
  | clone only | 26.00% |
  | real only | 24.44% |
  | hybrid | 25.12% |

  Synthetic data does not beat real overall, and adding it to real data does not help on average. Clone and hybrid do
  beat real for moderate-to-severe speakers and cross-corpus (SAP-1102).
- This matches our results: Higgs augmentation is null on Whisper and Nemotron (PLAN H, H2, re-voicing). The published
  Higgs gain lives in a subgroup and in cross-corpus transfer, not in same-speaker test WER.

### 1.3 Methods built for dysarthria that report real gains
| Work | Generator | How pathology is modelled | Result (TORGO unless noted) | Code |
|---|---|---|---|---|
| [Leung et al., Interspeech 2024, 2406.08568](https://arxiv.org/abs/2406.08568) | multi-speaker diffusion TTS (Grad-TTS family) | trained on dysarthric speech | beats prior dysarthric-ASR baselines when fine-tuning Whisper | not checked |
| [DARS, Wu et al., APSIPA 2025, 2603.01369](https://arxiv.org/abs/2603.01369) | Matcha-TTS (conditional flow matching) | rhythm predictor trained with normal-vs-dysarthric contrastive preferences, plus a dysarthric-style flow-matching condition | MCD 4.29; 54.22% relative WER reduction vs prior SOTA augmentation | not checked |
| [Wang et al., 2602.08696 (v3)](https://arxiv.org/abs/2602.08696) | **IndexTTS + LoRA** (codec LM) | separate timbre prefix and learnable patient-specific *pathology* prefix, disentangled with gradient reversal and VC counterfactuals; pathology can be put on healthy voices | partly replaces real data; adds speaker-diverse augmentation on top of real (Whisper-medium/large, leave-one-speaker-out) | [released](https://github.com/Mors20/Controllable-Dysarthric-Speech-Synthesis) |
| [Wagner et al., Interspeech 2025, 2505.12991](https://arxiv.org/abs/2505.12991) | **Parler-TTS fine-tuned** to mimic dysarthric speech, transcripts from an LLM | text-description control, x-vector personalisation | up to ~7% relative WER gain over personalised fine-tuning alone (SAP challenge, includes PD) | not checked |
| [Li et al., Interspeech 2025, 2505.14874](https://arxiv.org/abs/2505.14874) | voice conversion (UUVC) of healthy speech into dysarthric speech | converts **both speaker and prosody** | beats speed/tempo perturbation (Spanish, Italian, Tamil) | not checked |

Common thread: the methods that help model **rhythm/prosody and pathology explicitly**, apart from timbre. Higgs and
F5 zero-shot copy timbre well but regress the rest toward healthy speech.

### 1.4 Generation material, amount and quality (general ASR)
- **What text to generate:** phoneme-frequency-guided text selection (texts rich in the training set's common phonemes)
  beats random selection on 9 of 13 test sets, by up to 19.3% relative. Filtering the reference speech also helps
  ([Wang et al., 2608.26697](https://arxiv.org/abs/2608.26697)).
- **Diversity:** naive real+synthetic mixing often fails because synthetic speech is less diverse. Pitch augmentation
  and VC speaker augmentation did nothing. Jointly varying the other attributes gave up to 35% relative on LibriSpeech
  ([Ogun et al., 2503.08954](https://arxiv.org/abs/2503.08954)).
- **Synthetic-vs-real gap:** convolving synthetic audio with room impulse responses narrows the gap by reproducing the
  acoustic conditions of real recordings ([Labrak et al., SLT 2026 submission, 2606.29031](https://arxiv.org/abs/2606.29031)).
  This is cheap for us: the patient clips are room recordings, and Higgs output is clean studio audio.
- **Fine-tune, filter and evaluate the generator** rather than using it as it comes: large ASR gains on conversational
  Québec French ([Perrin et al., 2508.21631](https://arxiv.org/abs/2508.21631)).

### 1.5 Diffusion, flow-matching and GAN options (item 6)
- **Flow matching:** F5-TTS (large, zero-shot, cleans up speech, 1.1) and Matcha-TTS (small, trainable on little data,
  the DARS base). Fine-tuning a dysarthric-intelligibility converter on F5-TTS also exists
  ([2506.16127](https://arxiv.org/abs/2506.16127)), but it goes in the wrong direction for us.
- **Diffusion:** Grad-TTS-style models trained on dysarthric data (Leung 2024; also used for UA-Speech classification
  augmentation, [Springer 2026](https://link.springer.com/article/10.1007/s11517-026-03705-x)). DuTa-VC is a
  diffusion VC reported to help dysarthric ASR *(secondary, via search)*.
- **GAN:** the strongest GAN use here is as vocoders (HiFi-GAN family inside VITS/Matcha). VITS-based augmentation for
  dysarthric child speech is reported at 23% CER reduction ([Forum Acusticum 2026](https://dael.euracoustics.org/confs/fa2026/data/articles/000227.pdf), not read in full).
- **What could move into Higgs:** (a) an explicit pathology/rhythm condition, as in ProtoDisent's prefix or DARS's
  rhythm predictor; (b) RIR/room augmentation after generation; (c) selecting for pathology instead of intelligibility.
  (a) needs Higgs training code changes. (b) and (c) need none and can be tried first.

## 2. Candidate generators (fine-tunable, open weights)

Licence and fine-tune support come from the repos and listings found 2026-10-29 *(mostly secondary; verify before use)*.

| Candidate | Family | Fine-tune path | Why consider it | Risk |
|---|---|---|---|---|
| **IndexTTS-2 + LoRA** | codec LM | community LoRA ([risingore](https://github.com/risingore/index-tts-2-lora), ~50 clips); ProtoDisent code on IndexTTS | the only dysarthria method found with released code and pathology/timbre separation | licence to check; ProtoDisent's repo may target IndexTTS v1 |
| **F5-TTS** (fine-tuned) | flow matching | official fine-tune scripts | the diffusion-family option for item 6; well studied on TORGO | known to clean up speech (1.1); weights non-commercial *(secondary)* |
| **VoxCPM2** | tokenizer-free continuous LM | official LoRA and full fine-tune ([repo](https://github.com/OpenBMB/VoxCPM)); README says 5-10 min of audio adapts a speaker | official LoRA path; continuous output avoids codec artefacts | no dysarthria results found |
| **Matcha-TTS (DARS-style)** | small flow matching | train or fine-tune on patient data | explicit rhythm modelling, the strongest reported gains | DARS code not found; needs our own rhythm module |
| **Parler-TTS** | description-conditioned | fine-tuning supported | used on SAP including PD (our semg2 is PD) | small gain (~7% rel) |
| **VC of healthy speech** (kNN-VC / Seed-VC / UUVC) | voice conversion | none, or fine-tune on the patient | unlimited lexical content from LibriSpeech or TTS | needs prosody conversion too (Li 2025); our re-voicing (design B) was null |
| Qwen3-TTS / CosyVoice3 / Orpheus / Dots.TTS / MOSS-TTS | codec LM | official or community fine-tuning *(secondary)* | strong general cloning | no dysarthria evidence; the CosyVoice3 LoRA author reports worse WER after LoRA |

**Proposed shortlist for the benchmark: IndexTTS-2 + pathology LoRA, F5-TTS fine-tuned, VoxCPM2 LoRA.** Higgs stays as
the reference. These cover three generator families (codec LM, flow matching, continuous LM) and include the one
published method for disentangling pathology.

## 3. Proposed benchmark (to pre-register in PLAN before any run)

0. **Diagnose Higgs first (no training, ~2 GPU-h).** Compare our filtered Higgs clips with the patients' real clips on:
   - zero-shot ASR-judge WER;
   - speaking rate and pause ratio;
   - speaker similarity.

   If synthetic clips are much easier than real ones, this is the cleanup failure (1.1). That would explain the null
   and sets the selection criterion for the other models.
1. **Leakage guard (hard rule).** No generated sentence may match any val, test, oov, s2ov or s3ov sentence after text
   normalisation (lowercase, strip punctuation). A script checks this and fails the job before training.
2. **Generator gate per candidate.** Fine-tune on each patient's train clips only, generate 200 clips, and measure the
   judge WER gap to real clips, speaker similarity and rate. A candidate passes if it is at least as close to real as
   Higgs on two of the three.
3. **One factor at a time.** Fixed Whisper E2 recipe, seeds 1-5, metrics s1/s2 test, oov and s2ov_test. The factors:
   - generator: Higgs vs the shortlist;
   - fine-tuning strategy: zero-shot vs LoRA vs pathology-prefix;
   - text: patient set vs open-vocab/caregiver vs phoneme-guided generic vs conversational;
   - amount: 0.5x / 1x / 3x real;
   - post-processing: none vs RIR.
4. **Success rule:** the same interval rule as earlier studies (all seeds below the real-only mean, gap > seed sd).
   Subgroup reads (severity, OOV) are secondary only.

Cheapest first: step 0 and the RIR post-processing arm use existing Higgs clips and need no new generator.

## 4. Gaps
- No paper found on our patients' severity mix (mild PD, severe CP-like) with audio-visual targets.
- No head-to-head of modern fine-tunable TTS on dysarthric speech; the comparison in step 3 would be new.
- Seed-VC, Spark-TTS and Llasa: no dysarthria or fine-tuning evidence found in this search.
- The conversation-style data paper (PLAN item 7) is not the linked arXiv 2610.23117, which is test-time adaptation
  (item 8). The right link is still needed.
