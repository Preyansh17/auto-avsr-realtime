# Streaming Speech Recognition for Dysarthric Patient Speech

**Preyansh Agrawal** · 2026-08-10

---

## 1. Executive summary

This work builds a **real-time** speech recognizer for patients with dysarthric speech, and asks whether the
**video stream (lip movement) earns its cost** on this data.

Four model lines were built and evaluated. Two are deployable:

| System | Merged WER | Legal WER (cross-domain) | TTFT | Word lag | RTF |
| --- | --- | --- | --- | --- | --- |
| **Whisper large-v3 + SpecAugment**, full FT, 1.2 s chunks | **8.97%** | **6.56%** | 1.76 s | 1.61 s | 0.170 |
| **Nemotron 0.6B**, 210 epochs + speed perturbation | 10.76% | **6.56%** | **0.80 s** | **0.70 s** | **0.02–0.03** |
| AV Emformer, audio-only (streaming-selected) | — | 26.8% | — | — | ~0.2 |
| AV Emformer, audio-visual (streaming-selected) | — | 36.6% | — | — | ~0.2 |
| CarelessWhisper (causal), best config | 32.29% | — | 0.63 s | 0.05 s | 0.11 |

**Five conclusions:**

1. **Streaming is nearly free in accuracy terms; the cost is latency.** For Whisper the offline→streaming
   gap is **1.35pp** at the right decode point (8.97% vs 7.62%), not the ~4pp assumed for most of the
   investigation. For Nemotron the gap is **exactly zero** — streaming WER equals offline greedy WER on all
   9 domain × seed pairs, in every configuration tested.
2. **The two deployable models tie on accuracy where they are comparable, and differ ~2× in latency and ~7×
   in compute.** Whisper wins the merged test by 1.79pp; on the like-for-like cross-domain legal test both
   score **6.56% median**. The choice is therefore latency and cost, not accuracy.
3. **Video does not currently help.** Audio-only beats audio-visual in every architecture tested, including
   inside the audio-visual model itself (26.8% vs 36.6%). The audio-only systems are in an entirely
   different regime (7–11%).
4. **One merged model beats three per-domain specialists**, including on each specialist's own test set
   (merged→legal 6.56% vs legal→legal 10.38%). At this data scale the per-domain models are data-starved.
5. **Decode configuration mattered as much as training.** Re-tuning the decode point alone, with no
   retraining, cut Whisper's merged streaming error from 11.96% to 8.97% and halved its compute.

The dominant constraint throughout is **data scale**: ~314 training clips, with test sets of 40 / 30 / 10
clips. This drives the three-seed protocol, the three-way split, and the decision to treat the smallest
domain as non-decision-relevant.

---

## 2. Problem and research question

**Target population.** Patients with dysarthric speech (ALS and related conditions), where off-the-shelf ASR
degrades sharply. Data was collected across two separately-gathered domains, referred to throughout as
**legal** (larger, cleaner) and **legacy** (smaller, older, higher variance), plus a **merged** union.

**The use case is live captioning, not offline transcription.** This is the most consequential framing
decision in the work, because it makes latency a first-class metric rather than an afterthought: a caption
that is accurate but arrives three seconds after the word was spoken is not usable in conversation. Every
system is therefore characterized on both axes, and two distinct latency metrics are reported (§7).

**Research question, from the audio-visual speech recognition literature:** does the video stream help on
this data? Dysarthric speech is acoustically atypical, which is precisely the setting where the AVSR
literature predicts visual input should pay off most.

**Answer, on current evidence: no, not yet.** §8 gives the evidence, and the reasons to treat this as a
statement about the current configuration rather than about audio-visual ASR in general.

---

## 3. Systems built

### 3.1 The streaming audio-visual model

The starting point was an **offline** audio-visual recognizer: a Conformer encoder with CTC/attention beam
search over the whole utterance. That architecture cannot stream, because its attention is bidirectional over
the full input.

It was replaced with an **Emformer RNN-T transducer**, decoded chunk-by-chunk with both encoder state and
decoder hypothesis state carried across chunk boundaries.

- The pretrained model ships in a serialized inference-only form; converting it into a fine-tunable form was
  verified at **100% of parameters mapped with maximum numerical difference 0.0**, so fine-tuning starts from
  the published model rather than an approximation of it.
- Each decode step consumes a fixed window: past frames to satisfy the frontend's receptive field (trimmed
  from the features afterwards), plus a new segment and its right-context lookahead. Encoder state and the
  decoder's beam hypotheses persist across calls.
- **Algorithmic latency** of the configuration used is **1.44 s** — a 32-frame segment plus 4-frame right
  context at 25 fps. Real-time factor is ≈ 0.2 on a laptop CPU.
- **Streaming is file-based**: existing clips are consumed segment-by-segment as if arriving live, with
  incremental transcript output. There is no microphone or camera capture. This is deliberate — it keeps
  evaluation reproducible and makes latency measurable against a known ground truth.
- Two architecture variants are supported: one with pretrained weights (12 Emformer layers, dim 256, linear
  video frontend on 44×44 face crops) and one trained from scratch (20 layers, dim 128, Conv3d+ResNet18 on
  88×88 mouth ROIs).
- 24.7 M parameters for full fine-tuning; LoRA adapters train ~2.4% of parameters, with selectable scopes
  (encoder, predictor, joiner, fusion, video frontend, or all).

### 3.2 The audio-only systems

Three audio-only systems were built to answer the video question, each with its own streaming evaluator:

| System | Streaming mechanism |
| --- | --- |
| **Whisper large-v3** | SimulStreaming **AlignAtt** policy over an unmodified bidirectional Whisper. The encoder re-runs over a growing buffer; the position of the decoder's cross-attention decides which words are safe to commit. Knobs: segment length, frame threshold. |
| **Nemotron 0.6B** | Cache-aware FastConformer-RNNT. Encoder caches (channel and time) are carried across chunks by construction. Knob: attention context size. |
| **CarelessWhisper / WhisperRT** | Genuinely causal — Whisper's encoder attention is masked so frame *t* attends only to frames ≤ *t*, then retrained. O(1) cost per chunk, no re-encoding of history. |

The architectural distinction between the first and third explains their opposite result profiles.
**SimulStreaming changes only the decode policy on a frozen, full-strength model**: accuracy is preserved,
latency is floored by how long the policy must wait for context, and compute grows with the buffer. **
CarelessWhisper changes the architecture and retrains**: latency is solved structurally, but the
bidirectional pretrained representation is discarded and must be re-earned from limited data.

All four systems share one streaming-evaluation interface, one word-error-rate implementation, and one
latency instrument, so that results are directly comparable across systems.

---

## 4. Data and evaluation protocol

### 4.1 The split

All reported numbers use an independently audited three-way split (seed = 7) with per-domain source hashes
and a verified zero-overlap audit across all three domains:

| Domain | Train | Val (selection) | Test (held out) | Test reference words |
| --- | --- | --- | --- | --- |
| Legal | 238 | 30 | 30 | 183 |
| Legacy | 76 | 10 | 10 | ~40 |
| Merged | 314 | 40 | 40 | ~225 |

Clips average ~5.1 s. Video is lip-crop only; audio is stored alongside as waveform files. This split gave
the legacy domain its first honest train/validation/test partition; earlier work had honest splits for legal
and merged only.

### 4.2 Protocol design, and why

Three protocol decisions do most of the work in making these results trustworthy, and each was adopted in
response to a measured problem in preliminary results rather than on principle alone.

**Three seeds for every configuration, reported as median or mean with the spread visible.** Preliminary
single-run comparisons proved worthless at this data scale: one early configuration reported 41.7% error, but
repeated runs of the identical recipe spanned **~24 percentage points** (population: 41.7, 42.1, 47.0, 52.7,
57.4, 61.8, 66.3%), with an honest median of 50.3%. Re-evaluating a fixed checkpoint reproduces
bit-identically, so the variance is entirely in training, not evaluation. No single-seed number is presented
as a result anywhere in this report.

**A three-way split, so that checkpoint selection and reporting use disjoint data.** Selecting the best of
30 epochs on a validation set and then reporting on that same set is optimistic by **~4–5pp**: one Whisper
configuration read 8.1% under that procedure, 13.0% at true last-epoch, and 14.2% on an honest three-way
split. This is large enough to reverse conclusions, and did — both the ranking of two model sizes and the
sign of SpecAugment's effect on the largest model changed once selection was made honest.

**Pooled corpus word error rate** — total errors over total reference words, the standard convention —
rather than a macro-average of per-clip rates. The two differ by 1–3pp systematically, so mixing them across
systems makes a leaderboard invalid as a comparison.

### 4.3 Statistical sensitivity, and the legacy domain

Test sets are small, and the resulting sensitivity is stated up front: on legal, one word ≈ 0.55pp; on
merged, ≈ 0.44pp; on **legacy, one word = 2.5pp and one clip = 10pp**.

Legacy is consequently treated as **not decision-relevant**. Its apparent monotonic regression across the
Nemotron epoch sweep (20.0 → 22.5 → 25.0 → 30.0%) amounts to **four words**. Its offline seed range is
22.5pp, widening to 27.5pp under streaming, with the seed ranking inverting entirely — the best offline seed
(25.0%) is the worst streaming seed (50.0%). Configuration decisions therefore rest on merged (N=40) and
legal (N=30). Legacy is reported for completeness and labelled as such wherever it appears.

---

## 5. Results

### 5.1 Audio-visual Emformer

**Best results: audio-only 26.8%, audio-visual 36.6%** (legal domain, streaming decode).

**Checkpoint selection dominated every other factor on this model.** Selecting the final checkpoint was
originally the only method used. Two apparently reasonable alternatives were both shown to be invalid:
validation loss **anti-correlates** with word error rate here (the lowest-loss checkpoint emits empty
transcripts, an 88.9%-error blank-emission failure), and greedy whole-utterance validation error does not
track real streaming error. Selecting in the deployment condition — decoding every candidate epoch in actual
chunked streaming mode and selecting on a disjoint set — beat final-checkpoint selection by **17–42pp on all
six seeds tested** (one run improved from 83.6% to 37.2%). The 26.8% / 36.6% pair comes from this procedure.

**Visual input format was ablated rather than assumed.** Mouth-region crops reached **42.1%** against
**59.6%** for full-face crops under matched conditions. The face-crop route was closed: at 44×44 input
resolution a full-face crop leaves the mouth at roughly 12 px against roughly 44 px for a mouth crop.

**Also tested and rejected on this model:** checkpoint averaging (median 53.6% vs 50.5% — worse median and
wider spread); SpecAugment (tied at the shorter training length, worse at the longer); longer training (2850
steps beat 5700 by 10.7pp); learning-rate annealing combined with error-based selection (41.7% → ~60%); and
bf16 precision (~5pp worse than fp16-mixed).

### 5.2 Whisper large-v3 + SpecAugment — the accuracy line

**Best results: merged offline 7.62%, merged streaming 8.97%, legal cross-domain 6.56%.**

Recipe: full fine-tuning of encoder and decoder, SpecAugment on, learning rate 1e-5, 30 epochs, effective
batch 16, bf16. Decoding at 1.2 s segments with frame threshold 18.

**The published dysarthria recipe did not transfer.** Freezing the first five encoder layers with
augmentation — the approach reported for dysarthric speech in the literature — lost to full fine-tuning at
every model size tested: small frozen 29.2%, small full 23.8%, medium full 18.4%, turbo full with SpecAugment
12.4%, medium full with SpecAugment 11.9%, and large-v3 full with SpecAugment best of all. Full fine-tuning
beating partial freezing is the most robust finding in this work; it holds on Nemotron as well, by ~31pp.

**The one training lever that landed** was warmup steps 100 → **25** and weight decay 0.0 → **0.01**, both
inherited unexamined from an earlier low-rank recipe (100 warmup steps was ~17% of a ~600-step run):

| Domain | Offline mean (tuned vs baseline) | Streaming mean (tuned vs baseline) |
| --- | --- | --- |
| **Merged** | **7.62%** vs 8.22% | **11.96%** vs 17.04% |
| Legal | 6.38% vs 7.65% | 11.66% vs 14.39% |
| Legacy (not decision-relevant) | 21.67% vs 27.5% | 32.50% vs 35.0% |

Merged is a clean win: every tuned seed matches or beats the baseline's *best* seed (16.14%).

**The decode point was then re-swept, and the previously accepted setting did not survive.** The conclusion
that 0.6 s segments with frame threshold 25 was optimal had been measured on pre-tuning checkpoints.
Re-sweeping decode only, with no retraining:

| Decode point | Merged streaming WER (3 seeds) | Spread | TTFT p50 | Word lag p50 | RTF |
| --- | --- | --- | --- | --- | --- |
| 0.3 s / ft=25 | 12.11% | — | ~1.4 s | — | — |
| 0.6 s / ft=25 (previous) | 11.96% (11.66 / 8.07 / 16.14) | 8.07pp | ~1.5 s | ~1.6 s | 0.300 |
| **1.2 s / ft=18 (adopted)** | **8.97%** (8.97 / 8.97 / 8.97) | **0.00pp** | ~2.0 s | ~1.8 s | **0.170** |
| 3.0 s / ft=22 | 7.17% | — | ~2.7 s | — | — |

Four observations follow:

1. **Better accuracy at roughly half the compute**, for ~0.5 s more time-to-first-text. These configurations
   form a **Pareto frontier**: no point dominates another, so there is no single best setting without first
   stating which axis matters. A 2.4 s point *was* strictly dominated, and was discarded.
2. **1.2 s is a robustness point, not merely a better mean.** All three seeds produce identical error there,
   while configurations on either side show 5–8pp of seed spread. The seed scatter that had looked like
   training instability was largely a short decode buffer amplifying small differences between checkpoints.
3. **The offline-to-streaming gap is 1.35pp**, not the ~4pp assumed for most of the investigation. Most of
   the apparent streaming penalty was a decode-configuration artifact; what streaming actually costs here is
   latency, not accuracy.
4. **The 3.0 s row is a warning, not a win.** On ~5.1 s clips a 3.0 s segment is roughly two chunks per clip
   — offline decoding in disguise, at 2.7 s time-to-first-text. Lying on the Pareto frontier does not make an
   operating point meaningful.

The change generalizes across domains: merged 11.96 → 8.97%, legal 11.66 → 10.02%, legacy 32.50 → 30.00%
(each domain's own model on its own test set). Legal is clean — two seeds improve, one ties, none regress.
Legacy is directional only. A 1.8 s apparent improvement found while extending the grid was traced to seed
noise and withdrawn.

**The cross-domain result.** Until recently the Whisper results had only a diagonal — each domain's own model
evaluated on its own test set — whereas Nemotron trains once on merged and evaluates every domain
cross-domain. The two were therefore not structurally comparable. Decoding all three merged models against
the legal test set at the adopted operating point, with test membership pinned identical to the diagonal run:

| Seed | merged→legal streaming WER | CER |
| --- | --- | --- |
| 1 | 6.56% | 3.93% |
| 2 | 6.56% | 4.29% |
| 3 | 7.10% | 4.79% |
| **median / mean** | **6.56% / 6.74%** | |

Two findings:

- **Against Nemotron's matching cell this is a tie**, at 6.56% median for both systems (§6.1).
- **The merged model beats the legal specialist on legal's own test set**: 6.56% against 10.38% (medians,
  same 30 clips, same references). The reverse direction agrees — the legal model scores 7.93% on the merged
  test, also below merged's own diagonal. The consistent reading is that at this data scale the per-domain
  models are **data-starved**, and the merged model is the better model *everywhere*, including on each
  domain's home ground. This matches the earlier per-domain breakdown within merged, where the legacy portion
  scored 12.5–25.0% inside the merged model against 17.5–40.0% standalone, and the legal portion 4.92–6.56%
  inside merged against 6.56–8.74% standalone.

**Practical consequence: deploy one merged model, not a per-domain family.**

### 5.3 Nemotron 0.6B — the latency and cost line

**Best results: merged 10.76%, legal cross-domain 6.56% (best seed 5.46%), 0.80 s time-to-first-text,
0.70 s word-commit lag, real-time factor 0.02–0.03.**

Recipe: full fine-tuning, learning rate 1e-4 flat, weight decay 1e-3, warmup 200, effective batch 4,
bf16-mixed, **210 epochs**, SpecAugment on, **speed perturbation on**. Trained once on merged and evaluated
cross-domain.

**Epoch count was the dominant lever**, and its optimum proved split-dependent rather than a property of the
recipe — an earlier, less rigorous split had indicated a much lower sweet spot, and that did not hold:

| Epochs | Seed 1 | Seed 2 | Seed 3 | Median |
| --- | --- | --- | --- | --- |
| 90 | 20.63 | 21.08 | 20.63 | 20.63% |
| 120 | 10.76 | 16.59 | 14.80 | 14.80% |
| 150 | 13.00 | 13.00 | 13.90 | 13.00% |
| 180 | 10.76 | 12.11 | 13.90 | 12.11% |
| **210** | 13.00 | 11.21 | 11.21 | **11.21%** |
| 240 | 11.21 | 12.56 | 11.21 | 11.21% |

210 was chosen over 240 on cost and risk grounds; the two tie on median.

**Speed perturbation produced the final configuration**: merged 10.76% (−0.45pp) and legal 6.56% (−1.64pp),
beating its non-perturbed counterpart on all three seeds individually. Legacy regressed 5pp to 30.00%,
consistent with the monotonic legacy trend in §4.3 and not decision-relevant at N=10. Training cost roughly
doubles.

**Two structural findings:**

- **Streaming error equals offline greedy error exactly** on all 9 domain × seed pairs, in every
  configuration across the investigation. Chunked cache-aware decoding costs this model nothing in accuracy.
  This is the sharpest contrast with Whisper, where streaming is a policy layered onto a bidirectional model.
- **Beam search is unavailable in the streaming path** — the cache-aware carried-state decoder implements a
  merge path for greedy decoding only, so beam search is an offline ceiling reference rather than a
  deployable option. At the best configuration it now *loses* to greedy on merged (11.66% vs 10.76%) and on
  legal (8.20% vs 6.56%). On one seed, beam produced byte-identical output to greedy while doing 1.5–2× the
  compute.

**Chunk size is a real, non-monotonic lever with a sweet spot:**

| Attention context | Chunk | Legal WER | Merged WER | TTFT p50 | Word lag p50 | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| [70,13] (default) | 1.12 s | 33.33% | 32.16% | 1.01–1.10 s | 0.86–0.99 s | 0.02–0.03 |
| **[70,6]** | 0.56 s | 38.80% | 36.56% | **0.63–0.70 s** | 0.72 s | 0.034 |
| [70,1] | 0.16 s | 39.34% | 35.68% | 1.20–1.29 s | 1.57–1.74 s | 0.11 |
| [70,0] (causal) | 0.08 s | 44.26% | 47.14% | 2.38–2.47 s | 3.19 s | 0.20–0.21 |

Shrinking the chunk below roughly 0.5 s makes latency *worse*, not better, because per-chunk overhead comes
to dominate. The [70,6] setting buys ~35% better time-to-first-text for +4–5pp error, and has not yet been
re-tuned at the current epoch count.

**Also rejected:** a differential decoder/joint learning rate (28.25%); a learning-rate sweep with the
batch-size confound removed (no rate in {3e-5, 5e-5, 2e-4, 3e-4} beats 1e-4); and freezing the encoder
(67.03% against 35.68% for full fine-tuning).

### 5.4 CarelessWhisper — the sub-second probe

**Best result: 32.29% error at 0.63 s time-to-first-text and 0.05 s word lag.**

The motivation was to break the ~1.5 s word-lag floor with a natively causal model whose cost per chunk is
constant rather than growing with the buffer.

**The latency claim held immediately; the accuracy never did.** Zero-shot released checkpoints at 300 ms
chunks gave 0.63 s time-to-first-text, **0.05 s word lag**, 33 ms per chunk and real-time factor 0.11 — the
fastest numbers anywhere in this work — at 125–136% error.

| Configuration | Test WER | TTFT p50 | Word lag p50 |
| --- | --- | --- | --- |
| Zero-shot | 125–136% | ~0.65 s | ~0.10 s |
| LoRA, small | 56.95–66.82% (mean ~61.7%) | 0.63 s | 0.06 s |
| Full fine-tune, small | 35.87% | 0.62 s | 0.05 s |
| **Full fine-tune, small + SpecAugment, lr 5e-6** | **32.29%** (4-seed mean 35.65%) | 0.63 s | 0.05 s |

**The latency floor never degraded**: accuracy improved by roughly 90pp from zero-shot with time-to-first-text
and word lag essentially unchanged. Six further experiments were run and **none beat 32.29%** — multi-seed
confirmation, a finer learning-rate sweep, longer training (34.53%), larger backbones (large-v2 46.19%,
large-v3 36.77%), higher adapter rank (45–49%), and additional context blocks (worse on both axes). The
configuration is treated as a well-searched local optimum.

**The one knob that improved accuracy destroyed the point of the architecture:** a 1000 ms chunk reached
27.35% but pushed time-to-first-text to 2.04 s — worse than the SimulStreaming baseline it was meant to beat,
i.e. dominated on both axes. It was rejected rather than quoted. The causal architecture only wins where
short chunks are actually exploited.

Two secondary findings: **smaller is better here** — the validation-to-test gap grows with capacity (14.1pp
for small, 20.9pp for large-v2, 31.3pp for large-v3 on 314 clips) — and forced alignment succeeded on this
dysarthric corpus with **zero failures**.

**Status: closed.** At 32–34% this line is too far from 8.97% to be a deployment candidate. It is retained as
evidence that the sub-second regime is reachable, and as a measurement of what reaching it currently costs.
Note for any non-research use: the CarelessWhisper implementation is licensed CC BY-NC 4.0 (non-commercial).

---

## 6. Cross-system comparison

### 6.1 Accuracy

| Domain | Whisper large-v3 (1.2 s/ft=18) | Nemotron 210 + speed perturbation | Verdict |
| --- | --- | --- | --- |
| Merged (N=40), in-domain | **8.97%** | 10.76% | Whisper by 1.79pp |
| Legal (N=30), cross-domain | 6.56% median / 6.74% mean | 6.56% median / 6.56% mean | **tie** |
| Legacy (N=10) | 30.00% | 30.00% | not decision-relevant |

The legal row is like-for-like: the same merged-trained models, the same 30 held-out clips, the same
references, both decoded in streaming mode. The 0.18pp difference on means is well inside the value of a
single word on a 183-word test set.

**The verdict is therefore not split on accuracy.** Whisper is at least as accurate on both
decision-relevant domains. Nemotron's case rests on latency and compute.

### 6.2 Latency and cost

Merged domain, three-seed medians, measured against forced-alignment ground truth:

| Metric | Nemotron | Whisper | Advantage |
| --- | --- | --- | --- |
| Time-to-first-text from speech onset, mean | **0.80 s** | 1.76 s | 2.2× |
| Word-commit lag, mean | **0.70 s** | 1.61 s | 2.3× |
| Time-to-first-text p95 | **1.83 s** | 2.34 s | 1.3× |
| Word-commit lag p95 | **1.76 s** | 2.54 s | 1.4× |
| Real-time factor | **~0.02–0.03** | 0.170 | ~6–8× |

Both systems improved by a similar ~0.25–0.3 s when moved onto the neutral reference, so the biases largely
cancelled and **the ordering survived the correction** — which is the reason to trust it. The advantage is
~2× on means but narrows to **1.3–1.4× at the 95th percentile**; the honest claim is the pair, not the mean
alone.

Latency is **architecture-driven, not data-driven** — it is flat across domains regardless of error rate.

### 6.3 Deployment reading

- **Nemotron for interactive live captioning.** Sub-second mean lag, running ~40× faster than real time, so
  it is viable on modest hardware. It costs ~1.8pp of merged error relative to Whisper.
- **Whisper where accuracy dominates** and ~1.8 s lag is acceptable. Its decode point is a Pareto choice
  rather than a fixed setting: the same models reach 7.17% at 3.0 s segments, or 12.11% at ~1.4 s
  time-to-first-text.
- **One merged model in either case**, rather than per-domain models.
- **CarelessWhisper is not a candidate**, but it bounds the frontier: 0.05 s lag is achievable today at
  roughly 23pp of additional error.

---

## 7. Latency measurement

Latency is the metric most easily overstated, and its definition was refined three times before the numbers
in §6.2 were considered trustworthy.

**Two metrics are reported, not one.** **Time-to-first-text** (first output appears) and **word-commit lag**
(a word finishes being spoken → that word's text becomes visible). The second is what a user experiences.

**Refinement 1 — simulated real time.** An evaluation loop can feed chunks as fast as it likes, which would
allow text to be counted as "appearing" before its audio would have arrived in a live setting. Chunk *k*'s
audio is therefore modelled as existing only at `(k+1) × segment_duration`, with compute time added on top.

**Refinement 2 — anchor to speech onset, not file start.** Leading silence differs greatly between systems
(0.02–0.90 s for Whisper against 1.2–2.4 s for Nemotron), so anchoring at file start measures recording
padding as much as model behaviour. This asymmetry had made an earlier comparison favour the wrong system,
and correcting it also overturned an earlier verdict that chunk size was not a useful lever (§5.3).

**Refinement 3 — measure against forced-alignment ground truth, not the model's own timestamps.** Word lag
had been computed against each model's *predicted* word end times. Transducer models emit late, and as the
subtrahend a late reference makes measured lag look artificially small — the metric was partly grading
itself. This was confirmed independently before rebuilding: an attention context of [70,13] implies
13 × 0.08 s = **1.04 s** of mandatory lookahead, yet measured lag was ~1.00 s, which is physically
impossible.

The corrected metric uses Montreal Forced Aligner word end times, with hypothesis and reference aligned by
edit distance and **only exactly-matched words timed**. Both references are reported side by side. Forced
alignment covered **all 80 test clips with zero failures**, so there is no alignment-failure selection bias.

**What the neutral reference revealed** — two predictions were made in advance, and one was wrong:

1. **Medians improved rather than worsened** (merged lag 1.02 → 0.70 s). Forced alignment marks a word's full
   acoustic extent, and on elongated dysarthric speech the model commits before the word finishes.
2. **The bias lived in the tails**: p95 lag 1.36 → 1.76 s, p95 time-to-first-text 1.10 → 1.83 s.
3. **A ~0.4 s emission-eagerness difference between seeds** was exposed that the self-referential metric had
   read as flat (seed 1 at 1.08–1.25 s against seeds 2–3 at 0.63–0.85 s). Emission timing is a property of
   the trained checkpoint, not only of the architecture.
4. **The 1.04 s architectural-floor argument above was refuted** by the corrected measurement (lag
   0.65–0.93 s) and is recorded as withdrawn.

---

## 8. The video question

**Answer on current evidence: video does not help.**

- **Within the audio-visual Emformer**, under identical honest selection: audio-only **26.8%** against
  audio-visual **36.6%** — video costs roughly 10pp.
- **Across architectures**, audio-only systems occupy a different regime: **8.97%** (Whisper) and **10.76%**
  (Nemotron) on merged, against 36.6% audio-visual. Roughly 3–4× the error of the best audio-only system.
- **Visual input format matters more than expected**: changing crop type alone moved error by 17.5pp
  (mouth-region 42.1% against full-face 59.6%), indicating that the visual path is fragile rather than merely
  unhelpful.

**Why this should be read as a statement about the current configuration, not about AVSR in general:**

1. ~314 training clips is very little from which to learn a visual-to-acoustic mapping.
2. The visual frontend and the fusion module were inherited pretrained and **never adapted to dysarthric
   articulation** — precisely where lip motion is most atypical, and where the literature predicts the
   benefit should be largest.
3. Cross-domain streaming error reached 95.8% in the audio-visual configuration, indicating severe domain
   sensitivity in the visual path.

This is the top-priority open question (§10).

---

## 9. What did not work

Recorded so that these directions are not re-attempted without new reason.

**Whisper.** Averaging the weights of three seeds (lands exactly at the seed mean, 17.04%, worse than the
best single seed at 16.14%); speed perturbation (promising on one seed, reversed across three — mean 17.94%
against 17.04%, with the spread widening); learning-rate sweep (5e-6 → 22.42%, 2e-5 → 20.18%, against 1e-5's
11.66%); epoch-count sweep (20 → 25.11%, 45 → 16.14%); SpecAugment strength sweep (0.3 → 19.28%,
0.7 → 12.11%); train-time buffer truncation, aimed directly at the offline-to-streaming gap (p=0.5 → 14.35%,
p=0.3 → 13.45%, and the gap *widened* — a dead mechanism rather than a tuning problem); streaming-aware
checkpoint selection (selects the same checkpoint that offline validation already selects); beam search under
the streaming policy (greedy 17.49% against beam-2 31.39% and beam-5 28.70%, with hallucinated filler); and
Whisper medium as a cheaper base (16.59% streaming at real-time factor 0.175, against large-v3's 8.97% at
0.170 — worse accuracy *and* more forward passes, 361 against 190).

**Nemotron.** Differential decoder/joint learning rate; learning-rate sweep; encoder freezing; beam search in
the streaming path; offline beam search at the best configuration.

**Audio-visual Emformer.** Full-face crops; checkpoint averaging; SpecAugment; longer training; learning-rate
annealing with error-based selection; bf16 precision.

**CarelessWhisper.** Larger backbones; higher adapter rank; additional context blocks; longer training;
1000 ms chunks (better accuracy, but latency worse than the baseline being replaced).

---

## 10. Limitations and open questions

- **Test sets are small.** 40 / 30 / 10 clips. Differences below roughly 1pp on merged or legal are not
  resolvable, and the legacy domain cannot resolve a 2–3pp effect at all.
- **The legacy domain is not decision-relevant**, so all configuration choices rest on two domains.
- **The audio-visual results are on an earlier data partition** than the audio results, so the two are not
  strictly comparable. The conclusion that video does not help is robust to this — the gap is roughly 3–4× —
  but the exact audio-visual figures should not be quoted alongside the audio figures as if measured
  identically.
- **The cross-domain matrix is incomplete.** Whisper has not been evaluated on the legacy test set from a
  merged model, so the two systems' comparison tables are matched on legal but not on legacy.
- **The video conclusion is conditional** on an unadapted visual frontend and fusion module (§8). This is the
  single most important open question.
- **No clinical evaluation.** Everything reported here is error rate and seconds. Whether 8.97% at 1.8 s lag
  or 10.76% at 0.7 s produces the better caption for an actual patient is not answerable from these tables.
- **Single-site, two-domain data.** No claim is made about generalization beyond this collection.

---

## 11. Next steps, in priority order

1. **Give the video stream a fair test before concluding it cannot help.** Adapt the visual frontend and the
   fusion module to dysarthric articulation rather than inheriting them pretrained. This is the untested part
   of the research question, and the question the project set out to answer.
2. **Push the latency frontier where it is cheapest.** Nemotron's [70,6] attention context reaches
   0.63–0.70 s time-to-first-text for +4–5pp error, and has never been tuned at the current epoch count — the
   sweep that identified it predates the current recipe.
3. **Clinical evaluation**, converting the accuracy/latency trade-off into a usability judgement with actual
   users.
4. **More data, particularly for the smaller domain.** It is where a modest collection effort would buy the
   most, and the finding that per-domain models are data-starved suggests the merged model would improve too.
5. **Re-evaluate the audio-visual model on the current data partition**, and complete the cross-domain matrix,
   so that every number in the comparison tables is measured under identical conditions.

---

## Appendix A — Investigation timeline

| Period | Focus | Outcome |
| --- | --- | --- |
| Mid-June | Streaming port | Offline architecture replaced with Emformer RNN-T; verified exact transfer of pretrained weights; low-rank fine-tuning added |
| Late June | Audio-visual diagnostics | Unimodal fusion path corrected; crop-type ablation; validation loss shown to anti-correlate with error |
| Late June | Audio baselines built | SpecAugment harness plus Whisper and Nemotron pipelines |
| End of June | **Variance discovery** | ~24pp unseeded run-to-run spread invalidates single-run comparison; three-seed protocol adopted |
| Early July | Whisper size and freezing sweep | Full fine-tuning beats partial freezing at every model size |
| Early July | Nemotron brought up | Full fine-tuning beats freezing by ~31pp; streaming decode shown to match offline decode |
| Early July | **Protocol corrections** | Error-rate definition unified; selection bias removed via a three-way split; audio-visual model given deployment-condition selection (26.8% / 36.6%) |
| Mid-July | Generalization matrix; streaming Whisper | Train × evaluate matrix across domains; streaming Whisper brought to working accuracy |
| Mid-July | Latency instrumented | Time-to-first-text and word-commit lag defined and measured; all systems re-run on the audited split |
| Mid-July | CarelessWhisper | Sub-second latency confirmed; accuracy plateaus at 32.29%; line closed |
| Mid-July | Nemotron epochs; latency corrected | Epoch sweep; speech-anchored measurement overturns two earlier verdicts |
| Late July | Whisper training and decode tuning | Warmup/weight-decay win; decode re-sweep reaches 8.97% with zero seed spread |
| Late July | Nemotron on the audited split | Epoch sweep to 240; speed perturbation reaches 10.76% / 6.56% |
| End of July | **Forced-alignment latency** | Self-referential metric replaced; one advance prediction refuted; head-to-head comparison corrected |
| End of July | Cross-domain cell | Whisper merged→legal reaches 6.56%, tying Nemotron; merged model shown to beat per-domain specialists |

## Appendix B — Final configurations

**Whisper.** `openai/whisper-large-v3`; full fine-tuning of encoder and decoder; learning rate 1e-5; **warmup
25 steps**; **weight decay 0.01**; 30 epochs; batch 8 with gradient accumulation 2 (effective 16); bf16;
SpecAugment with time-mask probability 0.5, time-mask length 40, feature-mask probability 0.1, feature-mask
length 13. Decoding at 1.2 s segments with frame threshold 18.

**Nemotron.** `nvidia/nemotron-speech-streaming-en-0.6b`; full fine-tuning; learning rate 1e-4 flat; weight
decay 1e-3; warmup 200; batch 2 with gradient accumulation 2; bf16-mixed; **210 epochs**; SpecAugment on;
**speed perturbation on**. Trained on merged and evaluated cross-domain.

**SpecAugment settings** follow the dysarthria-tuned variant from the literature rather than the
general-speech defaults. The load-bearing change is **10 time masks instead of 2**: frequency-mask parameter
13 (against 27), 1 frequency mask (against 2), time-mask parameter 40 (against 100), 10 time masks (against
2), time-mask probability 0.5.

**Audio-visual Emformer.** fp16-mixed precision; 2850 steps; low-rank adapters on all scopes with rank 8,
alpha 16, dropout 0.05; batch 2 with gradient accumulation 4; learning rate 5e-4; no annealing, no
error-based selection, no checkpoint averaging; explicit seed.
