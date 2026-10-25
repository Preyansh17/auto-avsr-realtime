# Progress Deck — Build Spec

**Date assembled:** 2026-07-29
**Audience of the deck:** advisor / lab meeting (ASR-literate research peers)
**Target:** 15 slides, ~20 minutes, plus a 3-slide appendix
**Who builds it:** an AI agent (Claude Cowork), working **from this file alone**. No repo, cluster, or
web access is needed or wanted — every number, table, and chart datapoint is inline below.

---

## TASK FOR THE BUILDING AGENT

Produce a **`.pptx`** deck from this specification. Use the `pptx` skill.

**Mapping rules — follow exactly:**
1. **One slide per `## Slide N — Title` heading.** 15 content slides, then the 3 `## A1/A2/A3` appendix
   slides. Do not merge, split, reorder, or add slides.
2. **`Takeaway`** → the slide's headline/subtitle line, or a highlighted callout bar. It is the single
   sentence the slide must land; keep it on the slide, not in the notes.
3. **`Body`** → the slide's bullet text. Usable close to verbatim; you may tighten wording for line
   length, but **do not change any number, comparison direction, or hedge** ("directional only", "not
   decision-relevant", "small-N" and similar qualifiers are load-bearing — keep them).
4. **Tables** → real PowerPoint tables, not images of tables. Keep the bolding as marked; bold marks the
   winning or chosen value.
5. **Charts** → native PowerPoint charts where possible, using the explicit series data given in the
   `CHART DATA` blocks. Titles, axis labels, and highlight instructions are given per chart.
6. **`Speaker notes`** → the slide's **notes pane**, never visible slide text. These are for the
   presenter (the project's author) to speak from.
7. **`Design note` / `Diagram`** → layout guidance for you; do not render the guidance text
   itself onto the slide.
8. **Source citations** (italic lines at the end of each slide) → notes pane, appended after the speaker
   notes. Do not put them on the slide face.

**Hard guardrails:**
- **Do not invent, interpolate, round, or "fill in" any number.** Every figure the deck needs is in this
  file. If something appears to be missing, leave a visible `[TODO: missing]` placeholder and list it in
  your handback message — do not estimate it.
- **Do not add content from outside this file** — no web search, no repo reading, no background knowledge
  about Whisper/Nemotron/dysarthric ASR. The framing here is deliberate.
- **Do not soften or dramatize.** Several slides report negative or retracted results on purpose; that is
  the point of the deck, not an error to smooth over.
- **Do not add a "thank you", "questions?", agenda, or roadmap slide.** 15 + 3 is the whole deck.
- Em-dashes, `%`, `pp`, and `×` in the text are intentional; keep them.

**Theme (pick once, apply throughout):**
- Clean academic style. One accent colour for the two deployable models (Whisper, Nemotron) and a muted
  grey-tone for the parked lines (AV Emformer, CarelessWhisper); reuse those two consistently in every
  chart and table so the reader learns the coding on slide 5 and keeps it.
- Sans-serif throughout. Body text no smaller than 16 pt; table text no smaller than 12 pt.
- Every chart with a WER axis gets a "lower is better" axis label or annotation.
- 16:9.

**Handback:** report the file path, the slide count, any `[TODO: missing]` placeholders you inserted, and
any place you had to shorten body text substantially.

---

## CONTENT RULES (these are about the subject matter — respect them while building)

1. **No per-clip transcripts, references, or hypotheses anywhere in the deck.** The audio and video are
   real patient speech (dysarthric / ALS speakers). Aggregate WER, latency, and counts only. This spec
   contains none; do not add examples of what the model transcribed, even as illustration.
2. **Every WER in this spec is already a 3-seed median or mean.** Never present a single-seed number as a
   result. Where a spread matters it is given explicitly — keep it.
3. **`merged` is the headline domain** (40-clip test). `legal` (30 clips) is secondary. **`legacy`
   (10 clips ≈ 40 words) is explicitly not decision-relevant** — one word moves it 2.5pp. It appears in
   this deck only where it is labelled as such. Do not promote it to a headline anywhere.
4. **A stale table exists elsewhere in this project** (`5.5% legal / 2.6% merged`, single-seed, older
   ad-hoc split). It is **superseded** and deliberately absent here. Do not reintroduce it.
5. **Nothing about text-to-speech / speech synthesis.** Out of scope by request.
6. Glossary, in case you need to expand an axis label: WER = word error rate (lower is better);
   `pp` = percentage points; RTF = real-time factor (compute seconds per audio second, <1.0 is faster
   than real time); TTFT = time to first text; MFA = Montreal Forced Aligner; ft = frame threshold.

**Spec block format below:** each slide gives **Takeaway**, **Body**, **Table** / **CHART DATA**,
optional **Design note**, **Speaker notes**, and an italic source line.

---

## Slide 1 — Title

**Takeaway:** This is a 7-week, four-model investigation into real-time speech recognition for patients
with dysarthric speech.

**Body (title block):**
- **Streaming Audio-Visual Speech Recognition for Dysarthric Patient Speech**
- Progress report — 2026-06-11 → 2026-07-29
- 4 model lines · 3 evaluation domains · 3 seeds per configuration · 120 commits

**Design note:** plain title slide. Optional small footer strip with the four model names as a preview
of the deck's structure: `AV Emformer · Whisper large-v3 · Nemotron 0.6B · CarelessWhisper`.

---

## Slide 2 — Problem and the research question

**Takeaway:** For patients, a caption that is accurate but 3 seconds late is not usable — so this project
optimizes accuracy *and* latency together, and asks whether the video stream earns its cost.

**Body:**
- Target users: patients with dysarthric speech (ALS and related), where off-the-shelf ASR degrades badly.
- Use case is **live captioning**, not offline transcription → the metric is *WER and latency*, not WER alone.
- Two data domains, collected separately: **legal** (larger, cleaner) and **legacy** (small, older,
  higher-variance). A **merged** corpus combines both.
- Driving question inherited from the audio-visual literature: **does the video stream (lip movement)
  actually help on this data?**
- Constraint that shapes everything downstream: **~314 training clips.** Every conclusion has to survive
  the variance that a dataset this small produces.

**Speaker notes:** the small-data constraint is the reason for the 3-seed protocol on slide 4 and for
several of the corrections on slide 14 — worth planting here so those slides land as consequences rather
than digressions.

---

## Slide 3 — What was built

**Takeaway:** The offline recognizer was replaced with a genuinely streaming one, and three audio-only
baselines were built alongside it under a shared evaluation harness.

**Body:**
- **Architecture port:** offline Conformer + CTC/attention beam search → **Emformer RNN-T transducer**,
  decoded chunk-by-chunk with encoder and hypothesis state carried across chunks.
- **Pretrained weights preserved exactly:** TorchScript → eager-mode bootstrap verified at **100% of
  parameters mapped, maximum numerical difference 0.0**. Fine-tuning starts from the published model, not
  an approximation of it.
- **Streaming is file-based:** existing clips are consumed segment-by-segment as if arriving live, with
  incremental transcript output. No microphone/camera capture — this keeps evaluation reproducible.
- **Algorithmic latency of the ported config:** 1.44 s (32-frame segment + 4-frame right context @ 25 fps).
  RTF ≈ 0.2 on a laptop CPU.
- **Three audio-only baselines** built to answer "does video help": **Whisper large-v3** (SimulStreaming
  AlignAtt decode), **NVIDIA Nemotron 0.6B** (cache-aware FastConformer-RNNT), **CarelessWhisper**
  (natively causal Whisper).
- **Shared harness:** one streaming-eval interface, one pooled-WER metric, one latency instrument across
  all four models, so rows in the leaderboard are actually comparable.

**Diagram — build this, as native shapes (not an image).** Single left-to-right pipeline across the lower
half of the slide, seven boxes joined by arrows:

`clip (audio + video)` → `chunker: segment + right context` → two stacked parallel boxes
`video frontend` / `audio frontend` → `fusion` → `Emformer encoder` → `RNN-T decoder` → `incremental text`

Annotations:
- A bracket or caption under the `chunker` box: `1.44 s algorithmic latency`.
- A small loop arrow on `Emformer encoder` returning to itself, labelled `state carried across chunks`.
- Grey out or dash the `video frontend` box only if it fits the theme — the video path exists but does not
  currently help (slide 7). Do not add any text saying so here; slide 7 makes that point.

If shape-level layout proves unreliable, fall back to a clean horizontal row of labelled boxes with plain
arrows and keep the two annotations as text captions. Do not substitute a bulleted list for the diagram.

*Source: `README.md`, `asr_baselines/README.md`.*

---

## Slide 4 — Data and evaluation protocol

**Takeaway:** All headline numbers are on an independently audited three-way split, at 3 seeds, with the
test set never touched by model selection.

**Body:**
- **Audited split (`split_v1`, seed=7)**, with per-domain source hashes and a verified zero-overlap audit:

| Domain | Train | Val (selection) | Test (held out) |
| --- | --- | --- | --- |
| Legal | 238 | 30 | 30 |
| Legacy | 76 | 10 | 10 |
| Merged | 314 | 40 | 40 |

- **Three-way, not two-way:** checkpoint selection happens on val; reporting happens on a test set that
  selection never saw. (Earlier numbers in this project did not have this — see slide 14.)
- **3 training seeds for every configuration**, reported as median or mean with the spread visible.
- **Pooled corpus WER** (total errors / total reference words, the jiwer/NIST convention) for every model.
- **Test sets are small — sensitivity stated up front:** legal test = 183 reference words (1 word ≈ 0.55pp);
  merged test ≈ 225 words (1 word ≈ 0.44pp); **legacy test ≈ 40 words (1 word = 2.5pp, 1 clip = 10pp)**.
- Consequence: **legacy cannot carry a claim** and is reported for completeness only. Configuration
  decisions rest on merged (N=40) and legal (N=30).
- This was also the **first honest split legacy ever had** in the project.

**Speaker notes:** if asked why legacy is not chased — its entire apparent 20.0% → 30.0% regression across
the Nemotron epoch sweep amounts to four words. That was checked and formally closed rather than tuned against.

*Source: `README.md` (split_v1 section), `results/week_results_2026-07-07_2026-07-13.md`.*

---

## Slide 5 — Headline leaderboard

**Takeaway:** Two audio models are deployable today — Whisper is at least as accurate on both domains,
Nemotron is ~2× faster and ~7× cheaper — and both beat the audio-visual model by a wide margin.

**Table (streaming / deployable decode, `split_v1`, both models merged-trained, 3-seed medians):**

| Model line | Merged WER | Legal WER (cross-domain) | TTFT | Word lag | RTF |
| --- | --- | --- | --- | --- | --- |
| **Whisper large-v3 + SpecAugment**, full FT, 1.2 s chunks | **8.97%** | **6.56%** | 1.76 s | 1.61 s | 0.170 |
| **Nemotron 0.6B**, 210 epochs + speed perturbation | 10.76% | **6.56%** | **0.80 s** | **0.70 s** | **0.02–0.03** |
| AV Emformer, audio-only, streaming-selected | — | 26.8% | — | — | ~0.2 |
| AV Emformer, audio-visual, streaming-selected | — | 36.6% | — | — | ~0.2 |
| CarelessWhisper (causal), best config | 32.29% | — | 0.63 s | 0.05 s | 0.11 |

Notes to render as small print under the table:
- **Both legal numbers are now like-for-like**: same merged-trained checkpoints, same 30 held-out legal
  clips, same references. Whisper's cross-domain cell was filled 2026-07-30; before that the table was
  comparing Whisper's legal-*specialist* checkpoint against Nemotron's merged one.
- **Legal is a tie**: identical 6.56% medians; means 6.74% (Whisper) vs 6.56% (Nemotron) — 0.18pp, well
  inside one word on a 183-word test set. Do not present either model as the cross-domain winner.
- Whisper's merged 8.97% is its accuracy-optimal decode point; it is a Pareto curve, not a single number (slide 9).
- TTFT and word lag are measured against MFA ground truth and anchored to speech onset (slide 11).
- AV Emformer numbers are legal-domain and legal-trained; it was never re-run on merged after the audio
  baselines overtook it.
- CarelessWhisper's 32.29% is a favorable seed; its 4-seed mean is 35.65%.

**CHART DATA 1 — grouped horizontal bar chart.**
Title: `Streaming WER by model line (split_v1, 3-seed)`
X-axis: `WER % — lower is better`, range 0–40. Y-axis: category labels below.
Two series. Blank cells mean "not measured" — omit the bar entirely, do not plot zero.

| Category (Y label) | Series "Merged" | Series "Legal (cross-domain)" |
| --- | --- | --- |
| Whisper large-v3 (1.2s) | 8.97 | 6.56 |
| Nemotron 0.6B (210ep+SP) | 10.76 | 6.56 |
| CarelessWhisper (causal) | 32.29 | |
| AV Emformer, audio-only | | 26.8 |
| AV Emformer, audio-visual | | 36.6 |

Highlight: accent colour for the Whisper and Nemotron rows, muted grey for the bottom three.
The two Legal bars are deliberately equal (6.56 = 6.56) — do **not** nudge, offset, or re-round them to
look different; the tie is the finding. A light bracket or caption spanning the two Legal bars reading
`tie, like-for-like` is welcome.

*Sources: `README.md`, `results/nemotron_splitv1_epoch_sweep_2026-07-18.md`, `results/whisper_streaming_investigation_2026-07-09.md`, `results/week_results_2026-06-23_2026-07-01.md`.*

---

## Slide 6 — AV Emformer: the ported model

**Takeaway:** The streaming port works and trains, but its best honest number is ~27%, and inside it the
video stream makes things worse rather than better.

**Body:**
- 24.7 M parameters. Both **full fine-tuning** and **LoRA** (trains ~2.4% of parameters) supported.
- Best honest numbers, using **streaming-mode checkpoint selection** — candidate epochs decoded in actual
  chunked streaming mode and selected on a disjoint set:
  - **audio-only 26.8%**
  - **audio-visual 36.6%**
- Selection method mattered enormously: streaming-mode selection beat naive last-checkpoint by
  **17–42pp on all 6 seeds tested** (one run: 83.6% → 37.2%).
- **Visual input format was ablated, not assumed:** mouth-ROI crops **42.1%** vs full-face crops **59.6%**
  under matched conditions. The face-crop route was closed — a 44×44 input puts the mouth in ~12 px
  instead of ~44 px.
- Also tried and rejected on this model: checkpoint averaging (worse median, wider spread), SpecAugment
  (tie at the good step count, worse at the longer one), longer training (2850 steps beat 5700 by 10.7pp).
- Two selection metrics that looked reasonable were shown broken here: **validation loss anti-correlates
  with WER**, and the low-loss checkpoint emits empty transcripts.

**Speaker notes:** the 26.8 / 36.6 pair is the one to defend — both are honest, both come from the same
selection procedure, so the audio-vs-AV comparison inside this architecture is apples-to-apples.

*Source: `results/week_results_2026-06-23_2026-07-01.md` (§31), `results/results_summary.md`.*

---

## Slide 7 — "Does video help?" — answered

**Takeaway:** Not yet. Audio-only beats audio-visual in every architecture tested, including inside the
audio-visual model itself.

**Body:**
- **Within the AV Emformer**, with identical honest selection: audio-only **26.8%** vs audio-visual **36.6%**
  — video costs ~10pp.
- **Across architectures**, audio-only models are in a different regime entirely: **8.97%** (Whisper) and
  **10.76%** (Nemotron) merged, versus **36.6%** for the audio-visual model.
- Adding video is therefore not a small loss — it is roughly 3–4× the error of the best audio-only system.
- This is a *current-evidence* answer, and the likely reasons are stated rather than hidden:
  - ~314 training clips is very little to learn a visual-acoustic mapping from.
  - The visual frontend and fusion module were inherited pretrained and never adapted to dysarthric
    articulation, which is exactly where lip motion is most atypical.
  - Video quality/crop variance is a live cost: swapping crop type alone moved WER by 17.5pp.
- **Practical consequence:** effort was moved to the audio-only lines, which is where every result on the
  next slides comes from.

**CHART DATA 2 — two small side-by-side bar charts** (two separate charts, not one combined — the
comparison is only valid *within* each panel).

Left panel, title `Within the same architecture`. Y-axis `Legal WER % — lower is better`, range 0–40:

| Bar | Value |
| --- | --- |
| AV Emformer, audio-only | 26.8 |
| AV Emformer, audio-visual | 36.6 |

Right panel, title `Across architectures`. Y-axis `Merged WER % — lower is better`, range 0–40:

| Bar | Value |
| --- | --- |
| Whisper (audio) | 8.97 |
| Nemotron (audio) | 10.76 |
| AV Emformer (audio-visual) | 36.6 |

Use the accent colour for audio-only/audio bars and the muted grey for audio-visual bars in both panels,
so "grey = video included" reads at a glance. Caption under the pair: `Video costs ~10pp inside the AV
model, and the AV model is ~4× the error of the best audio-only system.`

*Source: `results/week_results_2026-06-23_2026-07-01.md`, `asr_baselines/README.md`.*

---

## Slide 8 — Whisper large-v3 + SpecAugment: the accuracy line

**Takeaway:** Full fine-tuning plus SpecAugment is the strongest accuracy recipe found, and revisiting two
inherited hyperparameters cut streaming error by nearly a third.

**Body:**
- Recipe: `openai/whisper-large-v3`, **full fine-tune** (encoder + decoder), SpecAugment on, lr 1e-5,
  30 epochs, effective batch 16, bf16.
- **The published dysarthria recipe did not transfer.** Green et al.'s freeze-first-5-layers approach lost
  to full fine-tuning at *every* model size tested (small, medium, turbo, large-v3). Full FT is the single
  most robust finding in the project — it also holds on Nemotron, by ~31pp.
- **Bigger is better here, but only under an honest split** — and SpecAugment's sign on large-v3 flipped
  twice before the split was fixed (see slide 14).
- **The one training lever that landed:** warmup steps 100 → **25** and weight decay 0.0 → **0.01**, both
  inherited unexamined from an old LoRA recipe:

| Domain | Offline mean (tuned → vs baseline) | Streaming mean (tuned → vs baseline) |
| --- | --- | --- |
| **Merged** | **7.62%** vs 8.22% | **11.96%** vs 17.04% |
| Legal | 6.38% vs 7.65% | 11.66% vs 14.39% |
| Legacy (not decision-relevant) | 21.67% vs 27.5% | 32.50% vs 35.0% |

- Merged is a **clean** win: every tuned seed matches or beats the baseline's *best* seed (16.14%) — not one
  lucky draw.
- Two other levers tried the same week were rejected after 3-seed checks: checkpoint soup (weight-averaging
  3 seeds lands exactly at the seed mean, worse than the best seed) and speed perturbation (looked like a
  win on one seed; reversed on three, with the spread exploding).

- **One model beats three specialists.** The merged-trained checkpoint is better on legal's own test set
  than the legal-trained checkpoint is: **6.56% vs 10.38%** (medians, same 30 clips, same references). The
  reverse direction agrees — legal-trained on the merged test scores 7.93%, also below merged's own
  diagonal. At this data scale the per-domain checkpoints are **data-starved**, and the single merged model
  is the better model *everywhere*, including on each domain's home turf.
- Practical consequence: **ship one merged model**, not a per-domain family. The per-domain diagonal, read
  alone, understates what a single merged model delivers.

**Speaker notes:** the useful methodological point is the asymmetry — the levers that *looked* promising on
one seed died, and the boring inherited-defaults check is what actually paid. Worth 15 seconds. The
merged-beats-specialists finding is newer (2026-07-30) and is the practical one: it removes any argument for
maintaining separate per-domain checkpoints. Caveat if asked: only the merged→legal cell has been filled on
the Whisper side; merged→legacy has not been run, so the cross-domain matrix is not closed.

*Source: `README.md` (2026-07-18 section), `results/whisper_streaming_investigation_2026-07-09.md`
(cross-domain cell, 2026-07-30).*

---

## Slide 9 — Whisper: streaming decode is a Pareto curve, not a setting

**Takeaway:** Re-sweeping the decode configuration alone — no retraining — cut merged streaming WER from
11.96% to 8.97% and halved compute, which showed most of the assumed "streaming penalty" was a
configuration artifact.

**Body:**
- Decode has two knobs: **segment length** (how much audio per step) and **frame threshold** (how eagerly
  the AlignAtt policy commits words).
- The previously accepted "0.6 s is optimal" conclusion had been measured on *pre-tune* checkpoints and did
  not survive the recipe change. Re-swept decode-only:

| Decode point | Merged streaming WER (3 seeds) | Seed spread | TTFT p50 | Word lag p50 | RTF |
| --- | --- | --- | --- | --- | --- |
| 0.3 s / ft=25 | 12.11% | — | ~1.4 s | — | — |
| 0.6 s / ft=25 (previous default) | 11.96% (11.66 / 8.07 / 16.14) | 8.07pp | ~1.5 s | ~1.6 s | 0.300 |
| **1.2 s / ft=18 (new default)** | **8.97%** (8.97 / 8.97 / 8.97) | **0.00pp** | ~2.0 s | ~1.8 s | **0.170** |
| 3.0 s / ft=22 | 7.17% | — | ~2.7 s | — | — |

- **1.2 s is a robustness point, not just a better mean:** all three seeds land on the identical WER, while
  configurations on either side show 5–8pp seed spread.
- It **generalizes** across domains — here each domain's *own* checkpoint on its *own* test set (the
  diagonal): merged 11.96 → **8.97%**, legal 11.66 → **10.02%**, legacy 32.50 → 30.00% (directional only,
  small-N). Label these `own-domain checkpoints` on the slide so they are not confused with slide 5's
  merged-trained legal number (6.56%), which is a different checkpoint on the same legal clips.
- **The 3.0 s row is a warning, not a win:** merged clips average ~5.1 s, so a 3.0 s segment is ~2 chunks per
  clip — offline decoding in disguise, with TTFT p50 of 2.7 s.
- **Two structural findings:**
  1. At 1.2 s the offline → streaming gap is **1.35pp** (8.97% vs 7.62%), not the ~4pp assumed for most of
     this investigation. **What streaming costs here is latency, not accuracy.**
  2. The seed scatter that looked like training instability was largely a short decode buffer amplifying
     small model differences — given a 1.2 s buffer, all three checkpoints score identically.

**CHART DATA 3 — XY scatter (Pareto frontier).**
Title: `Whisper decode: accuracy vs latency (merged, decode-only re-sweep)`
X-axis: `TTFT p50 (seconds)`, range 1.0–3.0. Y-axis: `Merged streaming WER % — lower is better`, range 6–14.
Connect the points with a light line to show the frontier shape.

| Point label | X (TTFT p50 s) | Y (WER %) |
| --- | --- | --- |
| 0.3s / ft=25 | 1.4 | 12.11 |
| 0.6s / ft=25 | 1.5 | 11.96 |
| **1.2s / ft=18 — chosen** | 2.0 | 8.97 |
| 3.0s / ft=22 | 2.7 | 7.17 |

Highlight: enlarge and accent-colour the `1.2s / ft=18` marker, label it `chosen operating point`.
Add a text callout on the `3.0s` marker: `degenerate — ≈2 chunks per 5.1s clip, i.e. offline in disguise`.

*Source: `README.md` (2026-07-19 section), `results/whisper_streaming_investigation_2026-07-09.md`.*

---

## Slide 10 — Nemotron 0.6B: the latency and cost line

**Takeaway:** A cache-aware streaming model reaches comparable accuracy at 6–8× less compute, wins the
cross-domain test outright, and pays literally zero accuracy penalty for streaming.

**Body:**
- Model: `nvidia/nemotron-speech-streaming-en-0.6b`, cache-aware FastConformer-RNNT. Full fine-tune,
  lr 1e-4, bf16-mixed, SpecAugment on.
- **Epoch count was the single biggest lever** — and its optimum is split-dependent, not a property of the
  recipe (an earlier ad-hoc split had suggested the sweet spot was much lower):

| Epochs | Seed 1 | Seed 2 | Seed 3 | Median |
| --- | --- | --- | --- | --- |
| 90 | 20.63 | 21.08 | 20.63 | 20.63% |
| 120 | 10.76 | 16.59 | 14.80 | 14.80% |
| 150 | 13.00 | 13.00 | 13.90 | 13.00% |
| 180 | 10.76 | 12.11 | 13.90 | 12.11% |
| **210** | 13.00 | 11.21 | 11.21 | **11.21%** |
| 240 | 11.21 | 12.56 | 11.21 | 11.21% |

- **Best configuration: 210 epochs + speed perturbation** — merged **10.76%**, legal **6.56%** (best seed
  5.46%, the best cross-domain number the project produced), legacy 30.00% (small-N, not decision-relevant).
  It beat its non-perturbed counterpart on all three seeds individually.
- 210 was chosen over 240 on cost and risk grounds, not because 240 was worse — they tie on median.
- **Streaming WER equals offline greedy WER exactly** on all 9 domain × seed pairs, in every configuration
  tested across the whole investigation. **Chunked cache-aware decoding costs this model nothing in accuracy.**
- **Beam search is not available in the streaming path**, and this was verified rather than assumed: NeMo
  raises an explicit `NotImplementedError` on partial-hypothesis merging for both beam algorithms. Offline
  beam is deployable-irrelevant — and at the best configuration it now *loses* to greedy (11.66% vs 10.76%
  merged, 8.20% vs 6.56% legal).
- Rejected after implementation: differential decoder/joint learning rate (28.25%, not competitive).

**CHART DATA 4 — line chart with markers.**
Title: `Nemotron: epoch count is the dominant lever (merged, 3-seed median)`
X-axis: `Training epochs` (categorical, in the order given). Y-axis: `Merged WER % — lower is better`, range 8–22.

| X (epochs) | Series "Median WER" | Series "Seed min" | Series "Seed max" |
| --- | --- | --- | --- |
| 90 | 20.63 | 20.63 | 21.08 |
| 120 | 14.80 | 10.76 | 16.59 |
| 150 | 13.00 | 13.00 | 13.90 |
| 180 | 12.11 | 10.76 | 13.90 |
| 210 | 11.21 | 11.21 | 13.00 |
| 240 | 11.21 | 11.21 | 12.56 |

Render "Median WER" as the solid accent line with markers. Render "Seed min"/"Seed max" as a light shaded
band between them (no visible lines, no legend entries of their own — label the band `per-seed range`).
Add one separate, differently-shaped marker at `X=210, Y=10.76`, labelled `+ speed perturbation (best)`.
Annotate `X=210` on the axis with `chosen`.

*Source: `results/nemotron_splitv1_epoch_sweep_2026-07-18.md`, `README.md` (2026-07-24 update).*

---

## Slide 11 — How latency is measured

**Takeaway:** Latency was the metric most easily faked, so it was rebuilt three times — simulated real
time, anchored to speech onset, and referenced to forced-alignment ground truth.

**Body:**
- **Two metrics reported, not one:**
  - **TTFT** — first text appears.
  - **Word-commit lag** — a word finishes being spoken → that word's text is visible. This is the metric a
    user actually experiences.
- **Correction 1 — simulated real time.** The eval loop can feed chunks instantly, which would let text
  "appear" before its audio would have arrived live. Chunk *k*'s audio is therefore treated as existing only
  at `(k+1) × segment_duration`, and compute time is added on top.
- **Correction 2 — anchor to speech onset, not file start.** Leading silence differed hugely between models
  (Whisper 0.02–0.90 s vs **Nemotron 1.2–2.4 s**), so a file-start anchor was measuring recording padding,
  not the model. This asymmetry is what made an earlier "Nemotron's TTFT is worse" claim wrong.
- **Correction 3 — reference MFA ground truth, not the model's own timestamps.** Word lag was being computed
  against the model's *predicted* word end times. RNN-T emits late, so a late reference makes measured lag
  look artificially small — the metric was partly grading itself. Replaced with **Montreal Forced Aligner**
  word ends; only exactly-matched words (hypothesis aligned to reference by edit distance) are timed.
- **Coverage:** MFA aligned **all 80 `split_v1` test clips, zero drops**, on dysarthric speech — so there is
  no alignment-failure selection bias in the latency numbers.
- **What the ground-truth reference revealed:**
  - Medians got **better**, not worse (merged lag 1.02 → 0.70 s) — MFA marks a word's full acoustic extent,
    and on elongated dysarthric speech the model commits before the word finishes.
  - The bias lived in the **tails**: p95 lag 1.36 → 1.76 s, p95 TTFT 1.10 → 1.83 s.
  - It exposed a **~0.4 s emission-eagerness difference between seeds** that the self-referential metric had
    read as flat — emission timing is a checkpoint property, not just an architecture property.
  - It **refuted** a previously argued architectural latency floor (a 1.04 s mandatory-lookahead argument);
    that claim is recorded as retracted.

**Speaker notes:** the honest framing is that one of the two predictions made before running this was wrong,
and the wrong one is in the write-up. That is the point of measuring against something the model does not control.

*Source: commit `b2f63f7`, `asr_baselines/mfa_timings.py`, `results/nemotron_splitv1_epoch_sweep_2026-07-18.md`.*

---

## Slide 12 — Latency head-to-head (MFA-referenced)

**Takeaway:** On a neutral reference, Nemotron is ~2× faster on means and ~6–8× cheaper to run; the
advantage narrows to 1.3–1.4× in the tails, which is the honest version of the claim.

**Table (merged domain, 3-seed medians, MFA reference):**

| Metric | Nemotron | Whisper | Advantage |
| --- | --- | --- | --- |
| TTFT from speech onset, mean | **0.80 s** | 1.76 s | 2.2× |
| Word-commit lag, mean | **0.70 s** | 1.61 s | 2.3× |
| TTFT p95 | **1.83 s** | 2.34 s | 1.3× |
| Word-commit lag p95 | **1.76 s** | 2.54 s | 1.4× |
| RTF | **~0.02–0.03** | 0.170 | ~6–8× |
| Merged WER | 10.76% | **8.97%** | Whisper by 1.79pp |
| Legal WER (cross-domain) | 6.56% | 6.56% | tie |

**Body:**
- Both models improved by a similar ~0.25–0.3 s when moved onto the MFA reference, so the biases largely
  cancelled — **the ordering survived the correction**, which is the reason to trust it.
- **The verdict is not split on accuracy.** Whisper wins merged by 1.79pp and ties legal at 6.56% median
  (means 6.74% vs 6.56%, inside one word). **Nemotron's case is latency and compute, not accuracy** —
  ~2× lower lag and ~6–8× lower compute, for ~1.8pp of merged WER.
- That is a cleaner trade than it looked a day earlier: until Whisper's cross-domain cell was filled
  (2026-07-30) the deck credited Nemotron with a 3.46pp cross-domain accuracy win, which was an artifact of
  comparing Whisper's legal-specialist checkpoint against Nemotron's merged one.
- Latency is **architecture-driven, not data-driven** — it is flat across domains regardless of WER.
- Deployment read: **Nemotron for interactive live captioning** (sub-second lag, runs ~40× faster than real
  time, so it is viable on modest hardware); **Whisper where accuracy dominates and ~1.8 s lag is acceptable.**

**CHART DATA 5 — grouped vertical bar.**
Title: `Latency head-to-head, MFA-referenced (merged, 3-seed median)`
Y-axis: `Seconds — lower is better`, range 0–3. Two series, four category groups.

| Category | Series "Nemotron" | Series "Whisper" |
| --- | --- | --- |
| TTFT mean | 0.80 | 1.76 |
| TTFT p95 | 1.83 | 2.34 |
| Word lag mean | 0.70 | 1.61 |
| Word lag p95 | 1.76 | 2.54 |

Add a horizontal reference line at `y = 1.0` labelled `1 s — interactive threshold`. Use the same accent
colours as chart 1 so Nemotron/Whisper stay visually consistent across the deck. The reference line is the
point of the chart: only Nemotron's two means fall below it.

*Source: commit `b2f63f7`, `results/nemotron_splitv1_epoch_sweep_2026-07-18.md`.*

---

## Slide 13 — CarelessWhisper: the sub-second probe

**Takeaway:** A natively causal Whisper does reach the sub-second latency regime, and it was pushed hard
before being closed — the latency claim held, the accuracy never did.

**Body:**
- Motivation: break the ~1.5 s word-lag floor with a Whisper variant retrained to be genuinely causal, so
  cost per chunk is **O(1)** instead of growing with the audio buffer.
- **The latency claim held immediately.** Zero-shot released checkpoints, 300 ms chunks: **TTFT p50 0.63 s,
  word lag p50 0.05 s, 33 ms per chunk, RTF 0.11** — the fastest numbers anywhere in this project. But WER
  was 125–136% (unusable without fine-tuning).
- **Fine-tuning progression** (`split_v1` merged, held-out test):

| Configuration | Test WER | TTFT p50 | Word lag p50 |
| --- | --- | --- | --- |
| Zero-shot | 125–136% | ~0.65 s | ~0.10 s |
| LoRA, small | 56.95–66.82% (mean ~61.7%) | 0.63 s | 0.06 s |
| Full FT, small | 35.87% | 0.62 s | 0.05 s |
| **Full FT, small + SpecAugment, lr 5e-6** | **32.29%** (4-seed mean 35.65%) | 0.63 s | 0.05 s |

- **The latency floor never degraded** through any of it — accuracy improved by ~90pp from zero-shot with
  TTFT and lag essentially unchanged. The architecture's selling point is real.
- **Six further experiments were run and none beat 32.29%**: multi-seed, finer LR sweep, longer training,
  larger backbones (large-v2 46.19%, large-v3 36.77%), higher LoRA rank, extra context blocks. Treated as a
  well-searched local optimum, not an untried lever.
- The one knob that did improve WER — a 1000 ms chunk (27.35%) — **pushed TTFT to 2.04 s**, worse than the
  Whisper baseline it was meant to beat. That defeats the entire purpose, so it was rejected rather than quoted.
- **Also two clean secondary findings:** smaller is better here (val→test gap grows with capacity:
  small 14.1pp → large-v2 20.9pp → large-v3 31.3pp on 314 clips), and MFA aligned this dysarthric corpus
  with zero drops.
- **Status: closed 2026-07-19.** 32–34% is too far from 8.97% to be a deployment candidate. Kept as evidence
  that the sub-second regime is reachable, and as a bound on what it currently costs.
- License note if the deck goes outside the group: the CarelessWhisper code is **CC BY-NC 4.0
  (non-commercial)**.

**Speaker notes:** this is the "negative result worth 30 seconds" slide. The interesting content is the
trade-off shape — 0.05 s lag is achievable *today*, it just costs ~23pp of WER — not the individual numbers.

*Source: `results/whisper_streaming_investigation_2026-07-09.md` (the earlier old-merge runs, best 34.08%);
the `split_v1` numbers in the table above (35.87% baseline, 32.29% best, 35.65% 4-seed mean, and the six
follow-up experiments) are from the 2026-07-18/19 CarelessWhisper working notes and are **not yet written
up in `results/`** — worth closing that gap before the deck is presented, since it is the one table here
that an advisor cannot trace to a committed document.*

---

## Slide 14 — Five corrections that changed reported numbers

**Takeaway:** Five methodology bugs were each found to have been moving the headline number, and every
leaderboard on the previous slides is post-correction.

**Table — render as the whole slide:**

| Correction | What was wrong | Before → after |
| --- | --- | --- |
| **Seeding** | Runs were unseeded and compared one-to-one, on ~300 clips | A 41.7% "headline" was a lucky draw from a **~24pp spread**; honest median **50.3%**. 3-seed protocol adopted project-wide |
| **Selection bias** | Best-of-30 checkpoints were selected on the *same* validation set that was then reported | Whisper large-v3 **8.1% → 14.2%**. Also reversed two substantive conclusions (medium vs large-v3; whether SpecAugment helps large-v3) |
| **Silent fp16 corruption** | A checkpoint conversion quietly halved precision; the ~0.002 weight difference read as benign noise | Streaming Whisper produced fluent nonsense for days. Fix: bit-exactness (max diff 0.0) is now **asserted** on every conversion, plus an end-to-end decode check |
| **WER metric mismatch** | Audio-visual rows used a macro-average of per-clip WER; audio rows used pooled corpus WER | The two families were **never comparable**. All rows recomputed on pooled WER (1–3pp lower); both metrics now reported |
| **Latency self-reference** | Word lag was measured against the model's own predicted word times | Merged lag **1.02 → 0.70 s** mean, **p95 1.36 → 1.76 s**; refuted a claimed architectural floor; exposed a hidden ~0.4 s inter-seed difference |

**Body (one closing line):**
- Cost of these: roughly two weeks. Value: the numbers on slides 5, 9, 10 and 12 are the ones that survived,
  and **15+ tuning levers are documented as closed with reasons**, so they are not re-attempted.

**Speaker notes:** do not present this as an apology slide. Three of the five corrections *reversed a
conclusion the project had already written down* — that is the argument for the protocol, and it is the
reason the remaining numbers are worth acting on.

*Source: `results/week_results_2026-06-23_2026-07-01.md`, `results/whisper_streaming_investigation_2026-07-09.md`, commits `55ba846`, `b2f63f7`.*

---

## Slide 15 — Status and next steps

**Takeaway:** Two configurations are deployable and reproducible today; the open work is the video stream,
the latency/accuracy frontier, and a clinical-side evaluation.

**Body — Where things stand:**
- **Whisper large-v3 + SpecAugment, 1.2 s / ft=18** — merged **8.97%** streaming, 7.62% offline,
  legal **6.56%** cross-domain. Best accuracy.
- **Nemotron 0.6B, 210 epochs + speed perturbation** — merged **10.76%**, legal **6.56%** cross-domain,
  **0.80 s TTFT / 0.70 s lag**, RTF 0.02–0.03. Best latency and cost.
- **They tie on cross-domain legal; the choice is Whisper's 1.79pp on merged against Nemotron's ~2× latency
  and ~7× compute advantage.** Not an accuracy-vs-accuracy decision.
- Both have a **committed config file and a one-command reproduction script**; both include their bit-exact
  conversion check.
- **Ship one merged model, not per-domain checkpoints** — the merged model beats each domain's specialist on
  that domain's own test set.
- Audio-visual line parked at 26.8% audio-only / 36.6% audio-visual, with video not yet earning its cost.

**Body — Next steps, in priority order:**
1. **Give the video stream a fair chance before concluding it cannot help.** The visual frontend and fusion
   module were inherited pretrained and never adapted to dysarthric articulation — that is the untested
   part of the "does video help" question, not the architecture.
2. **Push the latency frontier where it is cheapest.** Nemotron's `att_context_size=[70,6]` reaches
   ~0.63–0.70 s TTFT for +4–5pp WER and has never been tuned at the best epoch count — the sweep that
   found it predates the current recipe.
3. **Clinical-side evaluation.** Everything so far is WER and seconds. Whether 9% WER at 1.8 s lag or
   11% at 0.7 s is the better caption for an actual patient is not answerable from these tables.
4. **More data, especially legacy.** Legacy is formally retired as a decision input at 10 clips / ~40 words;
   it is the domain where a small collection effort would buy the most.
5. **Close the cross-domain matrix.** Whisper's merged→legacy cell has not been run, so the Whisper and
   Nemotron comparison tables are matched on legal but not yet on legacy.

**Design note:** two columns — "Deployable now" (left) and "Open" (right). The left column should read as
finished; the right as chosen, not leftover.

*Source: `asr_baselines/configs/{whisper,nemotron}_splitv1_best.yaml`, `README.md`.*

---

# APPENDIX (optional — build only if time allows or if the advisor asks for detail)

## A1 — Best configurations, verbatim

**Whisper** (`asr_baselines/configs/whisper_splitv1_best.yaml`): `openai/whisper-large-v3`, full FT
(encoder + decoder), lr 1e-5, **warmup 25**, **weight decay 0.01**, 30 epochs, batch 8 × grad-accum 2
(effective 16), bf16, SpecAugment `mask_time_prob 0.5 / length 40 / feature_prob 0.1 / feature_length 13`.
Decode: `segment_length 1.2`, `frame_threshold 18`. Confidence recorded per domain: merged **high**,
legal medium, legacy low.

**Nemotron** (`asr_baselines/configs/nemotron_splitv1_best.yaml`):
`nvidia/nemotron-speech-streaming-en-0.6b`, full FT, lr 1e-4 flat, weight decay 1e-3, warmup 200,
batch 2 × grad-accum 2, bf16-mixed, **210 epochs**, SpecAugment on, **speed perturbation on**. Trained on
merged, evaluated cross-domain. MFA and model-referenced latency recorded separately.

**SpecAugment settings** are the Green et al. dysarthria-tuned variant, not the LibriSpeech defaults: the
load-bearing change is **10 time masks instead of 2** (`freq_mask_param 13`, `n_freq_masks 1`,
`time_mask_param 40`, `n_time_masks 10`, `time_mask_p 0.5`).

## A2 — Closed negative levers (so they are not re-attempted)

**Whisper:** checkpoint soup (lands at the seed mean); speed perturbation (reversed on 3 seeds); LR sweep
(5e-6 and 2e-5 both worse than 1e-5); epoch sweep (20 and 45 worse than 30); SpecAugment strength sweep;
train-time buffer truncation (the gap *widened* — a dead mechanism, not a tuning problem); streaming-aware
checkpoint selection (picks the same checkpoint offline validation already picks); beam search in
SimulStreaming (greedy 17.49% vs beam-5 28.70% — hallucinated filler); Whisper medium as a cheaper base
(worse WER *and* more forward passes: 361 vs 190).

**Nemotron:** differential decoder/joint LR (28.25%); beam in the streaming path (unimplemented upstream);
offline beam at the best config (now loses to greedy); frozen-encoder training (67% vs 36% — full FT wins by ~31pp).

**AV Emformer:** face crops instead of mouth ROI (59.6% vs 42.1%); checkpoint averaging; SpecAugment;
longer training (5700 steps vs 2850); LR annealing plus WER-based selection (41.7% → ~60%); validation-loss
checkpoint selection (anti-correlates with WER).

**CarelessWhisper:** larger backbones; higher LoRA rank; extra context blocks; longer training;
1000 ms chunks (better WER, but latency worse than the baseline it was replacing).

## A3 — Engineering and cluster operations

- **31 SLURM templates** covering training, sweeps, streaming eval, checkpoint selection, label
  regeneration, MFA alignment; plus two one-command best-result reproducers.
- **Three separate Python environments** (Whisper / Nemotron / CarelessWhisper) — dependency conflicts
  between the toolkits made a shared environment unworkable.
- **Cluster GPU-utilization monitor auto-cancels jobs averaging <50% for 2h+**, which killed four long
  runs mid-training. Responses: raised dataloader workers, added `--resume-ckpt` (verified it truly resumes
  rather than silently restarting), and salvage-eval from the best surviving checkpoint.
- **A 4-arm training-speed A/B** identified checkpoint-every-5-epochs as **~18% faster** end to end; naive
  I/O accounting over-predicted the saving by ~2×, the rest being page cache.
- Storage discipline: Lightning checkpoints are 7 GB each (Adam moments); ~670 GB of scratch reclaimed
  during the Nemotron campaign.
- **Patient-data safety:** a standing rule that any per-clip reference/hypothesis dump is public-repo-unsafe
  by default. One near-miss (2026-07-13) was caught before anything reached the remote and the history was
  rewritten. **This is why no example transcriptions appear in this deck.**

---

## Traceability

Every number above comes from one of: `README.md`; `results/whisper_streaming_investigation_2026-07-09.md`;
`results/nemotron_splitv1_epoch_sweep_2026-07-18.md`; `results/week_results_2026-06-23_2026-07-01.md`;
`results/week_results_2026-07-02_2026-07-06.md`; `results/week_results_2026-07-07_2026-07-13.md`;
`results/week_results_2026-07-14_2026-07-17.md`; `asr_baselines/configs/*.yaml`; or a cited commit.
Per-slide sources are given in italics at the end of each slide.
