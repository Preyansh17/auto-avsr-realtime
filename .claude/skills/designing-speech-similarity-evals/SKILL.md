---
name: designing-speech-similarity-evals
description: Designs and reports experiments that compare synthetic (TTS / voice-cloned) speech against real speech. Use when planning or changing an evaluation experiment, choosing test sentences, picking similarity metrics, building a listening test or demo page, or when the user asks "how is this being measured", "do we have ground truth for this", or "how will we compare with real speech".
---

# Designing speech similarity evaluations

A claim that synthetic speech sounds like the real speaker needs real audio of the same text to compare against. Design for that from the start.

## Steps

1. **Keep to the user's stated scope.** Use exactly the sample size, speakers and conditions the user gave (e.g. "sample size 5-10, exclude median-matched, binnge and M04 only, only for the data we have"). Do not add excluded conditions back or include speakers who lack data. Write the scope at the top of the experiment entry in PLAN.md.
2. **Pick sentences that have ground truth.** For each speaker, list the transcripts that have real recordings and choose the test sentences from that list. If the target text has no recording, build it from sentences or segments that do have recordings (e.g. stitch "what is it for" and "i did not get your point" so the stitched text has a real counterpart). Every synthetic clip must be paired with the real clip of the same text and speaker.
3. **Label anything that has no pair.** If a comparison has no real reference, say so in the plan and the report, and use only a reference-free metric, named as such. Never report a "similarity to real speech" number for clips that have no paired recording.
4. **Justify each metric before using it.** For each metric, state in one line which property it measures (speaker identity, prosody/pace, intelligibility). Drop metrics that do not measure the claim. WER (word error rate) measures intelligibility, not voice similarity, so do not use it as similarity evidence.
5. **Keep showcase sentences out of the reference audio.** A demo or example sentence must not appear in the reference clip used to generate it. Check this for every example in the artifact before handing it over, not just the new ones.
6. **Before running, show the design:** the scope, the sentence list with its paired real-clip paths, the metrics with what each measures, and any unpaired items. Then run.
7. **Before handing over a listening test or blind comparison, state its composition.** In the same message that gives the user the test, say how many clips or trials there are, which are real recordings and which are synthetic, which is the anchor/reference and which are probes, which model or condition produced each synthetic clip, how clips are ordered or randomized, and what the user should judge in each trial. If the test is blind, give these facts as counts and structure without revealing which clip is which. Put the per-clip answer key in a separate file and give its path. The user should not have to ask "are these all synthetic?" after finishing.

## Verify

- Sample size, speakers and conditions match what the user stated.
- Every synthetic clip used for a similarity claim has a paired real clip of the same text.
- Unpaired comparisons are labeled and use only reference-free metrics.
- Each metric has a one-line statement of what it measures; WER is not presented as similarity.
- No showcased sentence appears in its generation reference audio, checked across the whole artifact.
- Any listening test handed to the user came with its composition (real vs synthetic, anchor vs probe, condition per clip, ordering, what to judge), and the answer key is kept separate if the test is blind.
