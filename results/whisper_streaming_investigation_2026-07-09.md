# Streaming Whisper investigation — handoff notes (2026-07-09)

**Status: RESOLVED (session 2, same day).** Root cause: the converted OpenAI-format checkpoint was corrupt — saved in fp16 with values that don't bit-match the source HF weights. A fresh reconversion (bit-exact fp32) fixed everything: stock openai-whisper decodes the smoke clip correctly, and the full SimulStreaming streaming run emits the exact reference transcript incrementally. See "Session 2 findings" and "Resolution" below. Sections above the session-2 marker are the original (now historical) session-1 handoff.

---

## Why this happened

Nemotron and the AV Emformer both got real streaming-mode evals this session (`asr_baselines/nemotron_streaming_eval.py`, `slurm/select_best_streaming_epoch.sbatch`). Whisper had no streaming equivalent — its encoder is non-causal/bidirectional over a fixed 30s window, architecturally not a streaming model. Researched the actual state of the art (not guessed):

1. **SimulStreaming** (UFAL, 2025) — algorithmic wrapper using the **AlignAtt** policy (reads encoder-decoder attention to know how much of the buffer is safe to decode). No retraining needed, works on any existing Whisper checkpoint. ~5x faster than the older LocalAgreement-based `whisper_streaming`. Chosen as the first thing to try — cheapest, reuses our best existing checkpoint.
2. **CarelessWhisper / WhisperRT** (arXiv 2508.12301, Aug 2025 / updated Apr 2026) — turns Whisper into a genuinely causal streaming model via LoRA fine-tuning + causal attention masks. Real retraining, architecturally correct, not attempted this session.
3. Naive fixed-window chunking — known broken, not attempted.

User approved building option 1. This file documents that attempt.

---

## What's deployed and where

- **Env**: `/scratch/pa2753/envs/simulstream` — torch 2.5.1+cu121, torchaudio 2.5.1+cu121 (matched), transformers 5.13.0, librosa, tiktoken, soundfile, openai-whisper, ffmpeg (via conda-forge, `<8`).
- **Code**: `/scratch/pa2753/third_party/SimulStreaming` (cloned from `github.com/ufal/SimulStreaming`, MIT license) and `/scratch/pa2753/third_party/convert_hf_to_openai.py` (community HF→OpenAI-format checkpoint converter, from `github.com/hitz-zentroa/whisper-lm`). **Not on `/home`** — moved there deliberately after hitting `/home`'s inode quota (see Environmental hazards below).
- **Converted checkpoint**: `/scratch/pa2753/experiments/whisper_asr/whisper_largev3_specaug_3way_full_seed1/openai_format/large-v3-patient-ft.pt` (6.17GB fp32). Source: `whisper_largev3_specaug_3way_full_seed1/best` — our actual best Whisper config this whole investigation (honest three-way-split WER 5.46% legal_only, 2.64% merged; see `results/week_results_2026-06-23_2026-07-01.md` §30).
- **Test clip**: `/scratch/pa2753/experiments/whisper_asr/smoke_test_clip.wav`, extracted from legal_only test clip `86_Do you know what they think_26-300_repeat1_20251029_102913.mp4`. Reference: **"do you know what they think"**. Confirmed this exact clip decodes correctly (verbatim match) via the standard offline HF pipeline (`val_hyps.tsv` for this model).
- **Diagnostic scripts** (all under `/scratch/pa2753/`, reusable): `verify_conversion.py`, `verify_mel.py`, `verify_pos.py`, `verify_encoder.py`, `extract_wav.py`.

---

## Bugs found and fixed along the way (all resolved)

1. **Auto-mode safety classifier blocked initial deployment** of third-party code to the cluster — needed explicit user confirmation ("yes") before cloning/running SimulStreaming + the converter script.
2. **torchaudio/torch CUDA mismatch**: `pip install torchaudio` without `--index-url` pulled a build targeting a different CUDA than our installed torch (cu121) → `OSError: libcudart.so.13: cannot open shared object file`. Fixed with `--index-url https://download.pytorch.org/whl/cu121`, matching the same class of bug hit earlier this session with other envs.
3. **`TMPDIR` not set** → pip's internal build/download temp files went to the login node's small default `/tmp` → `No space left on device` (not a real `/scratch` quota issue — `/scratch` had 239T free). Fixed by exporting `TMPDIR=/scratch/pa2753/.tmp`.
4. **`/home` inode quota exceeded** mid-session (`touch` failed even after freeing ~180 files) — pre-existing, not caused by this session's work specifically, but the SimulStreaming clone's ~180 files were enough to tip it over. Fixed by moving `third_party/` from `/home/pa2753/auto-avsr-realtime/` to `/scratch/pa2753/` entirely. **`/home` is still at/near quota** — a standing hazard for any future work that writes many small files there.
5. **Checkpoint naming collision (real bug, patched)**: SimulStreaming's CLI wrapper (`simul_whisper.py`) strips the directory and `.pt` extension from `--model_path` before calling `load_model()`. If the resulting bare name matches one of openai-whisper's ~10 recognized stock model names (we'd named our file `large-v3.pt` → stripped to `"large-v3"`), the loader takes the download/checksum-validation branch and — on checksum mismatch (guaranteed, since ours is finetuned) — **silently re-downloads and uses the stock pretrained weights instead of our finetuned checkpoint**, with only a warning printed, no error. Fixed two ways: (a) renamed our checkpoint to `large-v3-patient-ft.pt` (not a recognized stock name), (b) patched `simul_whisper.py` to pass the full absolute path through to `load_model()` when the stripped name isn't in the stock registry, so this class of bug can't recur for other custom checkpoint names either.
6. **`ffmpeg` binary missing**: openai-whisper's own `.transcribe()` convenience method shells out to an `ffmpeg` binary via subprocess (separate code path from the streaming CLI, which uses `librosa` directly — that's why the streaming smoke test didn't hit this but a direct `.transcribe()` call did). Installed via conda-forge into the `simulstream` env.
7. **`n_mels` mismatch in `.transcribe()`**: the convenience wrapper defaults to `n_mels=80` (correct for older Whisper sizes) but large-v3 needs 128. Silently produces a shape-mismatch crash if not overridden. Worked around by calling the lower-level `log_mel_spectrogram(..., n_mels=model.dims.n_mels)` API directly instead of the wrapper.

All of the above are confirmed fixed and are not the cause of the remaining problem below.

---

## The unresolved blocker

After all fixes above, the actual transcription output is consistently garbage/hallucinated:
- First real streaming run: `"Good night, I think."`
- Second run (after checkpoint-loading fix): `"Good night, Jason."`
- Both for a clip whose correct answer, confirmed via the standard HF pipeline, is **"do you know what they think"**.

### Isolation testing performed (in order)

1. **Weight-level comparison** (`verify_conversion.py`, job 13175977): compared `conv1`, decoder token embedding, encoder layer-0 `q_proj` between the HF checkpoint and the converted OpenAI-format checkpoint. All match within 0.0005–0.002 absolute difference — consistent with expected bf16→fp32 rounding noise (the model was trained in `bf16-mixed`), **not** a conversion bug.
2. **Mel spectrogram comparison** (`verify_mel.py`, jobs 13177441/13178438): compared SimulStreaming's own `load_audio()` (ffmpeg via subprocess) + `log_mel_spectrogram()` against HF's `WhisperFeatureExtractor`, same wav file. Near-identical: max diff 0.0146, mean diff 0.00008.
3. **Positional embeddings + remaining tensors** (`verify_pos.py`, job 13181224): encoder positional embedding matches **exactly** (0.0 diff), decoder positional embedding, `ln_post`, `conv2` all match within the same tiny bf16-noise range as step 1.
4. **Full encoder forward pass comparison** (`verify_encoder.py`, job 13180654 — first attempt, job 13178988, timed out at 15min on CPU with no output; GPU rerun succeeded): fed the **same** mel input (from the HF feature extractor) through both `HF WhisperEncoder` and SimulStreaming's `AudioEncoder`, both in eval mode, both fp32. **Result: real divergence** — max abs diff 15.18, mean abs diff 0.222, against an output std of only ~0.78. This is not rounding noise.
5. **First decoder step**: fed the (diverged) SimulStreaming encoder output into the decoder after the start-of-transcript token sequence. Top prediction: `<|endoftext|>` at logit 26.96, next-best (`,`) at only 7.87 — an enormous, unusually confident margin toward immediate termination. This is the behavioral signature of a decoder receiving essentially uninformative/degenerate cross-attention input, not just noisy-but-usable input.
6. **Source inspection**: read SimulStreaming's vendored `model.py` (`AudioEncoder`, `MultiHeadAttention`, `ResidualAttentionBlock`). Architecture appears to be a faithful reproduction of vanilla openai-whisper — standard `conv1→gelu→conv2→gelu→+positional_embedding→N×ResidualAttentionBlock→ln_post`, standard scaled QKV attention (SDPA path disabled by default, falls back to manual softmax attention). No obviously-wrong code spotted by inspection alone. Note: some custom dtype-casting `LayerNorm`/`Linear`/`Conv1d` subclasses from vanilla openai-whisper are **commented out** in this vendored copy — not yet confirmed whether this matters (shouldn't, since everything here runs in fp32 throughout, but not ruled out).

### Root cause: not found. Leading untested hypotheses

- Some numerical instability/amplification specific to bf16-trained weights run through 32 encoder layers with subtly different op-ordering/intermediate-precision than HF's implementation — "equivalent" code can still compound differently over 32 residual layers, and this checkpoint is a **full fine-tune** (not frozen/LoRA), so weights have moved further from the original pretrained init than a lightly-adapted model would.
- Some remaining config/tensor mismatch not yet checked: attention bias/mask defaults, `n_audio_ctx` value consistency, LayerNorm `eps`, or the commented-out dtype-casting classes mattering after all.
- Not yet tried: layer-by-layer trace (compare block 0 output, block 1 output, ... block 31 output between HF and SimulStreaming) to find exactly which layer the divergence starts at and whether it's sudden (one bad layer) or gradual (compounding noise).

---

## Options for a new session

1. **Keep digging** — layer-by-layer trace through all 32 encoder blocks to localize exactly where divergence starts. More GPU jobs, more time, no guarantee of a clean fix at the end (could be a genuine, hard-to-fix numerical sensitivity).
2. **Abandon SimulStreaming, try CarelessWhisper/WhisperRT instead** (arXiv 2508.12301) — the causal-LoRA route. Different codebase, different failure modes, but architecturally the "correct" answer for true streaming Whisper, and this repo already has LoRA infrastructure (currently only wired to the AV Emformer).
3. **Stop here** — keep Whisper's offline honest numbers (5.46%/2.64%, `results/week_results_2026-06-23_2026-07-01.md` §30) as its contribution to the investigation, and don't pursue streaming Whisper further. Nemotron and the AV Emformer already have real streaming numbers; Whisper's role in this investigation was always as the non-streaming reference point.

No decision was made before this handoff — user asked for a summary to hand off to a new session rather than picking a direction.

---

## Session 2 findings (2026-07-09, later same day)

Picked up option 1 (keep digging). Major reframe: **SimulStreaming is exonerated — the bug reproduces with stock openai-whisper, offline, no streaming involved.**

1. **Code diff, vendored vs stock** (`diff` of SimulStreaming's vendored `model.py` against the pip-installed `whisper/model.py`): differences are cosmetic only — commented-out dtype-casting subclasses (irrelevant in fp32), kv-cache keyed by `cache_id` strings instead of module objects, `use_sdpa=False` (manual attention path, math identical), pos-embedding sliced `[:x.shape[1]]` (no-op at full length), `return_layer_results` hook added. `qkv_attention` math is standard. Vendored encoder ≈ stock encoder.
2. **Vendored `load_model` uses `load_state_dict` strict=True** — load succeeded, so no silently dropped/missing keys. Rules out the random-init-layers hypothesis.
3. **Decisive three-way test** (`verify_root_cause.py`, jobs 13181930 ffmpeg-crash / 13182418 complete):
   - **A) HF `generate()` in the simulstream env (transformers 5.13.0), fp32**: `' do you know what they think'` — **correct**. HF side is sane in this env; transformers v5 is not the culprit.
   - **B) Stock openai-whisper `decode()` with the converted checkpoint, offline, fp32**: `'Good night, Jason.'` — **same garbage as the streaming runs**. Not SimulStreaming, not streaming, not AlignAtt.
   - **C) Exhaustive all-1259-tensor comparison** (HF state dict renamed via the converter's own mapping vs the converted `.pt`): zero key mismatches, worst max-abs-diff 0.0019 (bf16 noise). The `.pt` file faithfully contains the HF-loaded values **under the converter's renaming**. Caveat: this check is circular w.r.t. mapping *semantics* — if a rename sent a tensor to the wrong slot, both sides would agree and C would still pass.
4. **`config.json` checked**: standard large-v3 dims; `apply_spec_augment: true` is training-only (and A decodes fine, so not it). Preprocessor config standard (128 mels, hop 160, 30s chunks).

**Current state of the mystery**: same weights (C), same mel input, both fp32/eval, near-identical code (1) — yet HF encoder and openai-whisper encoder outputs diverge (max 15.18 vs output std ~0.78, from session-1 `verify_encoder.py`). Remaining suspects: converter mapping semantics that C can't see (same-shape slot swaps — q/k/v/out are all d×d and mutually swappable; encoder vs decoder blocks same shapes), or some architectural assumption baked into openai-whisper code that this HF checkpoint violates.

5. **Layer-by-layer trace** (`verify_layers.py`, job 13182534): divergence present from the embed stage already (max 0.40 vs output std 0.72 — far above fp32-identical compute), grows gradually through the blocks (no single bad layer), and explodes at block 20 where large-v3's famous outlier channels appear (HF activation std jumps 0.30 → 5.10; max diff ~984 there). Interpretation: small weight-level noise, amplified by an outlier-heavy fine-tuned encoder.
6. **The tell**: step C's 0.0019 weight diff should have been exactly **zero** (the converter just renames tensors from the same `from_pretrained` load). Nonzero diff meant the old `.pt` did not faithfully contain the HF weights. File size confirmed it: old `.pt` was **3.09GB — pure fp16** (`Counter({'torch.float16': 1259})`), despite session 1 recording it as "6.17GB fp32". Session 1's weight spot-checks read the 0.0005–0.002 diffs as benign "bf16 training noise"; they were actually the corruption itself.
7. **Old-file provenance not fully explained**: old ≠ `fp16(new)` (1258/1259 tensors differ) and old ≠ `fp16(bf16(new))` (1259/1259 differ) — so it isn't a clean dtype roundtrip of the current `best` weights; the underlying values differ by ≤0.002. Possibly converted from marginally different source weights or via an extra cast chain. Left unresolved deliberately — forensics stopped once the fix was verified.

## Resolution

- **Fresh reconversion** (`reconvert_and_test.py`, job 13194413) in the simulstream env: new `.pt` is **bit-exact vs HF (max diff 0.0)**, fp32, 6.17GB.
- **Offline stock-whisper decode with new ckpt**: `'do you know what they think'` — correct.
- **Full SimulStreaming streaming smoke test** (job 13194920, `--comp_unaware --never_fire`, 1.2s segments): incremental emissions `" do you"` → `" know what they"` → `" think"` — exact reference transcript, with word timestamps. **Streaming Whisper works.**
- Sensitivity control: bf16-roundtripped HF weights still decode correctly via HF `generate` — the model isn't generically fragile to ~1e-3 weight noise; the old file's specific corruption was what broke it (exact mechanism of its badness not chased further).

**Artifacts:**
- Good checkpoint: `/scratch/pa2753/experiments/whisper_asr/whisper_largev3_specaug_3way_full_seed1/openai_format/large-v3-patient-ft-v2.pt` (fp32, bit-exact, verified end-to-end).
- Corrupt one quarantined as `large-v3-patient-ft-CORRUPT-do-not-use.pt` (same dir) — safe to delete.
- New reusable scripts on cluster: `/scratch/pa2753/verify_root_cause.py`, `verify_layers.py`, `reconvert_and_test.py` (+ matching `.sbatch` files, which now export `PATH`/`LD_LIBRARY_PATH` into the simulstream env for ffmpeg and `HF_HOME=/scratch/pa2753/hf_cache`).

**Lesson for future conversions**: after any checkpoint format conversion, (a) assert bit-exactness against the source state dict (max abs diff must be 0.0, not "small"), and (b) run an end-to-end offline decode of a known clip before debugging anything downstream.

## Batch streaming eval (session 3, same day — DONE)

Built `asr_baselines/whisper_streaming_eval.py` — Whisper counterpart to
`nemotron_streaming_eval.py`: one utterance = one stream (fresh decoder/KV state per
file via `online.init()`), audio fed in 1.2s segments through SimulStreaming's
AlignAtt policy (frame_threshold=25, greedy, `--never_fire` since no CIF model exists
for large-v3), per-chunk wall latency + RTF, shared `asr_baselines.metrics` normalizer.
Test-set membership pinned to the exact clips of the offline honest-split evals by
reading each run's `val_hyps.tsv` as the manifest. Deployed as a package copy at
`/scratch/pa2753/wse_pkg/` (NOT `/home` — inode quota blocked the scp) with sbatch
templates `/scratch/pa2753/whisper_streaming_eval{,_merged}.sbatch`.

Also converted the merged-trained median seed (`whisper_largev3_specaug_3way_merged_seed3`,
the offline 2.64% checkpoint) to OpenAI format with the now-mandatory bit-exact assert
(max diff 0.0, fp32): `.../merged_seed3/openai_format/large-v3-patient-ft-merged-s3.pt`.

**Results (jobs 13202883/13202884, L40S, segment=1.2s, frame_threshold=25):**

| Train / eval | Offline WER | Streaming WER | CER | RTF | ms/chunk |
| --- | --- | --- | --- | --- | --- |
| legal-trained seed1 / legal test30 | 5.46% | **9.29%** | 5.89% | 0.215 | 232 |
| merged-trained seed3 / merged test40 | 2.64% | **3.96%** | 3.33% | 0.189 | 202 |
| legal-trained seed1 / merged test40 (cross-domain bonus) | — | 7.93% | 6.86% | 0.178 | 190 |

Takeaways:
- Streaming costs Whisper +3.8pp (legal) / +1.3pp (merged) over its offline numbers —
  the expected AlignAtt partial-context penalty, nowhere near the catastrophic
  streaming degradation the AV Emformer showed early on.
- **Streaming Whisper is the best streaming system in the whole comparison by a wide
  margin**: 9.3%/4.0% vs AV Emformer audio-only 26.8% (legal) and Nemotron 33.3%/31.7%.
- Latency caveat for any writeup: SimulStreaming is pseudo-streaming — it recomputes
  the encoder over a growing (≤30s) buffer each 1.2s segment, so per-chunk cost
  (~200ms avg here) grows with utterance length, vs Nemotron's true carried-state
  ~26-31 ms/chunk. Both are comfortably real-time (RTF ~0.2 vs ~0.026) on an L40S.
- Single-seed numbers (median offline seeds). The known ~seed-variance caveat applies
  before leaderboard-grade claims; a 3-seed streaming pass is cheap (~2 GPU-min/seed)
  if needed.
- Env note: `tqdm` had been silently resolving from `~/.local` all along; installing
  under `PYTHONNOUSERSITE=1` placed it properly into the simulstream env.
- Data note: the label-CSV mp4 paths are video-only lip crops — audio lives in sibling
  `.wav` files (same convention as `patient_audio.load_waveform`); the eval script
  handles this.

Hyp TSVs: `.../3way_full_seed1/streaming_hyps.tsv`, `.../3way_full_seed1/streaming_hyps_mergedtest.tsv`,
`.../3way_merged_seed3/streaming_hyps.tsv`.

## Segment-length sweep + perceived-latency metrics (session 3 continued)

Added simulated-real-time latency instrumentation to `whisper_streaming_eval.py`
(chunk k's audio only exists at (k+1)*segment seconds in a live deployment; text from
it appears at that + compute — valid while RTF<1):
- **TTFT**: stream start → first text visible (includes each clip's leading silence).
- **Word commit lag**: word finished being spoken → its text visible (per-word, includes
  words only released by the end-of-stream flush).

Sweep results (job 13207666, L40S, frame_threshold=25, greedy):

| Segment | Set | WER | RTF | TTFT p50 | word lag mean / p50 / p95 |
| --- | --- | --- | --- | --- | --- |
| 1.2s | legal test30 | 9.29% | 0.20 | 2.59s | 1.98 / 1.89 / 2.75s |
| 1.2s | merged test40 | 3.96% | 0.18 | 2.59s | 1.85 / 1.83 / 2.61s |
| 0.6s | legal test30 | 13.11% | 0.31 | 1.97s | 1.85 / 1.65 / 3.21s |
| 0.6s | merged test40 | 6.17% | 0.30 | 1.97s | 1.56 / 1.49 / 2.21s |
| 0.3s | legal test30 | 8.74%* | 0.57 | 1.69s | 1.72 / 1.53 / 3.22s |
| 0.3s | merged test40 | 9.25% | 0.56 | 1.97s | 1.44 / 1.37 / 2.06s |

*legal test30 has only 183 ref words (1 word ≈ 0.55pp), so the 0.3s "win" over 1.2s is
one-word noise; merged (227 words) shows the real WER trend: 4.0% → 6.2% → 9.3%.

Takeaways:
- Latency floor is ~1.7-2.0s TTFT / ~1.4-1.5s word lag regardless of how short the
  segment gets — AlignAtt's hold-back (frame_threshold=25 = 0.5s) plus its need for
  ~1s of committed acoustic context dominate below 0.6s segments. Halving 0.6→0.3
  buys ~0.1-0.3s of latency for +3.1pp merged WER and doubled compute (RTF 0.30→0.56).
- Per-chunk compute barely drops with segment size (~217→163ms) — encoder recompute
  over the growing buffer dominates, so short segments burn compute fast.
- Best live operating point so far: **0.6s segments — words appear ~1.5-1.6s after
  being spoken (p95 ~2.2s), 6.2% merged WER, RTF 0.30**.
- The next latency lever is `frame_threshold` (25 frames = 0.5s hold-back), not
  segment length: dropping it to ~12 cuts 0.26s off every emission at zero compute
  cost, in exchange for more premature commits. Not yet swept.

Hyp TSVs per point: `streaming_hyps_seg{0.6,0.3}.tsv` in each run dir (1.2s = the
original `streaming_hyps*.tsv`).

## frame_threshold sweep (session 3 continued, job 13213744)

Tested the other latency knob at the 0.6s-segment sweet spot: `frame_threshold`
25 (default) / 18 / 12 (frames, 0.02s each — AlignAtt's hold-back from buffer end).

| frame_threshold | Legal WER | Merged WER | Merged word lag mean/p50/p95 |
| --- | --- | --- | --- |
| 25 | 13.11% | 6.17% | 1.56 / 1.49 / 2.21s |
| 18 | 13.11% | 6.61% | 1.46 / 1.41 / 2.11s |
| 12 | 13.66% | 7.49% | 1.38 / 1.35 / 2.05s |

**Not worth it.** Halving the nominal hold-back (25→12 frames, i.e. 0.5s→0.24s) only
bought ~0.14-0.2s of real word-lag reduction (p50 1.49→1.35s) — far less than the naive
0.26s prediction, meaning most of the ~1.4-1.6s lag floor is NOT the AlignAtt hold-back;
it's decode/context latency intrinsic to the model. Cost: +1.3pp merged WER (6.2%→7.5%),
legal actually got worse too (13.1%→13.7%). frame_threshold=25 (the SimulStreaming
default) is already close to optimal for this checkpoint — no further latency to
extract from this knob or from segment length.

**Conclusion for real-time use**: ~1.4-1.6s word lag is a structural floor for
AlignAtt-wrapped Whisper on this checkpoint at reasonable WER, not fixable by tuning
`segment_length` or `frame_threshold`. Getting meaningfully below ~1s would need
genuinely causal streaming (CarelessWhisper/WhisperRT, the causal-LoRA route flagged
as option 2 in the original handoff), not more pseudo-streaming parameter search.

Hyp TSVs: `streaming_hyps_seg0.6_ft{25,18,12}.tsv` in each run dir.

## Nemotron streaming latency, for comparison (session 3 continued)

While answering "what's Nemotron's equivalent latency", found the README's
"Nemotron streaming, full finetune | 33.3% | 31.7%" row was **mislabeled**: traced
the sbatch that produced those two numbers (`nemotron_ft_12895525/529/530.out`,
checkpoints `nemotron_fullft_3way_legal_seed3` / `nemotron_fullft_3way_merged_seed1`)
and it calls `nemotron_eval.py` (offline `transcribe()`), not
`nemotron_streaming_eval.py`. True cache-aware streaming Nemotron WER+latency on the
honest 3-way splits had never actually been run — only smoke-tested on an old,
different checkpoint at 30 clips (`nemotron_stream_eval_1273248*.out`, ~34-36% WER,
not the reported 33.3/31.7 config).

Built `nemotron_streaming_eval_latency.py` — Nemotron counterpart to
`whisper_streaming_eval.py`'s TTFT/word-lag instrumentation. Audio-consumed-so-far
computed as `streaming_buffer.buffer_idx / streams_length * manifest_duration`
(fraction of this utterance's feature frames processed times its real duration) —
avoids hardcoding the model's frame-stride constant, robust across streaming_cfg.

**Word-level lag: blocked by a real NeMo bug, not our code.** Tried enabling
`compute_timestamps=True` via `model.change_decoding_strategy()` to get word_offsets
(mirroring Whisper's per-word timestamps). Both of NeMo's greedy RNNT decode paths
reject it during cache-aware streaming's carried partial-hypothesis state:
- `loop_labels=True` (default, job 13231047): `hyp.merge_()` crashes —
  `AttributeError: 'dict' object has no attribute 'extend'` (`hyp.timestamp` is a
  dict when timestamps are on; the merge code assumes a list).
- `loop_labels=False` (job 13233294): `_greedy_decode_blank_as_pad_loop_frames`
  explicitly raises `NotImplementedError("`partial_hypotheses` support is not
  supported")`.

Confirmed both fail independently — this is a genuine gap in NeMo's cache-aware
streaming + timestamp support, not fixable from the eval script. Fell back to
TTFT-only (job 13234666, `want_word_ts` hardcoded off with the reasoning logged
inline in the script).

**Real streaming results** (job 13234666, L40S, native streaming_cfg
`chunk_size=[105,112] shift_size=[105,112]`):

| Checkpoint / eval | Offline WER | Streaming WER | RTF | ms/chunk | TTFT mean/p50/p95 |
| --- | --- | --- | --- | --- | --- |
| legal seed3 / legal test30 | 33.33% | **33.33%** | 0.026 | 26.2 | 2.77 / 2.19 / 3.31s |
| merged seed1 / merged test40 | 31.72% | **32.16%** | 0.019 | 18.9 | 3.11 / 3.30 / 4.42s |

TTFT n=25/30 and n=35/40 — some utterances never produced text before their first
tracked emission point within the window counted (short/silence-leading clips).

Takeaways:
- **Near-zero streaming WER penalty for Nemotron** (33.33%→33.33%, 31.72%→32.16%,
  i.e. +0 to +0.44pp) — contrast with Whisper's +1.3 to +3.8pp penalty. Makes sense
  architecturally: Nemotron's FastConformer encoder is cache-aware/causal by
  construction, so its "offline" decode was never peeking at future audio the way
  Whisper's bidirectional encoder does; there's little streaming penalty to incur.
  This means the WER numbers in the old mislabeled README row were *directionally*
  fine even though the methodology label was wrong — the real bug this surfaced was
  the missing latency numbers, not a wrong WER.
- **But TTFT is worse than Whisper's despite ~10x lower per-chunk compute.**
  Nemotron: 19-26ms/chunk, RTF 0.02-0.03. Whisper (1.2s segments): ~200ms/chunk, RTF
  ~0.2. Yet Nemotron's TTFT (2.2-4.4s) is comparable to or worse than Whisper's
  (1.7-2.6s at 1.2s segments, 1.0-2.0s at 0.3-0.6s). Reason: Nemotron's native
  streaming chunk size (`chunk_size=[105,112]` encoder frames) is a large fixed
  block — algorithmic latency (how much audio must accumulate before ANY output)
  dominates over compute latency, same concept `scripts/benchmark_latency.py`
  calls "algorithmic latency = (segment+right_context)/fps" for the AV Emformer.
  Whisper's short, tunable `segment_length` gives it a real latency advantage that
  raw per-chunk speed numbers alone would hide.
- Nemotron's `chunk_size`/`shift_size` weren't swept here (kept at pretrained
  defaults, as the original `nemotron_streaming_eval.py` docstring recommends) —
  a chunk-size sweep analogous to Whisper's segment-length sweep is a clean
  follow-up if Nemotron's latency profile needs deeper characterization.

Deployed at `/scratch/pa2753/nemo_eval_pkg/asr_baselines/nemotron_streaming_eval_latency.py`.
Hyp TSVs: `nemotron_fullft_3way_{legal_seed3,merged_seed1}/streaming_hyps_latency.tsv`.

## Three-way latency comparison (session 3, final)

| Model | Mode | Legal WER | Merged WER | RTF | TTFT p50 | Word-lag p50 |
| --- | --- | --- | --- | --- | --- | --- |
| Whisper large-v3 (1.2s) | streaming | 9.29% | 3.96% | 0.18-0.20 | 2.6s | 1.8-1.9s |
| Whisper large-v3 (0.6s) | streaming | 13.11% | 6.17% | 0.30-0.31 | 2.0s | 1.5-1.7s |
| Nemotron | streaming | 33.33% | 32.16% | 0.019-0.026 | 2.2-3.3s | unavailable (NeMo bug) |
| AV Emformer, audio-only | streaming | 26.8% | — | — (see `scripts/benchmark_latency.py`) | — | — |

Whisper wins on both WER and (with 0.6s segments) latency, despite far higher
compute-per-chunk than Nemotron — Nemotron's compute efficiency doesn't translate
into a latency win because its native chunk size is large and untuned here.

## Nemotron chunk-size (att_context_size) sweep (session 3, final)

Followed up on "reduce Nemotron's chunk size" to see if the untuned-chunk-size
theory above actually buys lower TTFT. Raw `chunk_size`/`shift_size` args to
`encoder.setup_streaming_params()` don't work directly for this model (passing
them without both set raises `TypeError: unsupported operand type(s) for -: 'int'
and 'NoneType'` in NeMo's own code) — this FastConformer only supports a small set
of **discrete `att_context_size` presets**, found via
`model.encoder.att_context_size_all`: `[[70,13], [70,6], [70,1], [70,0]]` (first
number = left context, fixed at 70; second = right-context/lookahead steps).
Inspected each preset's resulting frame-level chunk size (`subsampling_factor=8`,
`window_stride=0.01s`): `[70,13]`→112 frames (1.12s, the pretrained default),
`[70,6]`→56 frames (0.56s), `[70,1]`→16 frames (0.16s), `[70,0]`→8 frames (0.08s,
fully causal, zero lookahead).

Added `--att-context-size "left,right"` to `nemotron_streaming_eval_latency.py`
(routes to `setup_streaming_params(att_context_size=[...])`, the correct lever;
kept the old `--chunk-size`/`--shift-size` flags for direct override if ever
needed). Swept all 4 presets on both checkpoints/test sets (job 13238618, L40S):

| att_context | Chunk duration | Legal WER | Merged WER | RTF | TTFT p50 |
| --- | --- | --- | --- | --- | --- |
| [70,13] (pretrained default) | 1.12s | 33.33% | 32.16% | 0.02-0.04 | 2.19-3.30s |
| [70,6] | 0.56s | 38.80% | 36.56% | 0.03 | 2.19-2.74s |
| [70,1] | 0.16s | 39.34% | 35.68% | 0.10 | 2.19-2.34s |
| [70,0] (fully causal) | 0.08s | 44.26% | 47.14% | 0.18 | 2.18-2.26s |

**Chunk-size reduction is not a useful lever for Nemotron.** Shrinking the chunk
14x (1.12s→0.08s) moved TTFT p50 by at most ~1s (2.19-3.30s → 2.18-2.26s) while WER
degraded badly and monotonically (32-33%→44-47% at fully causal). Two things going
on: (1) TTFT here is dominated by something other than encoder lookahead window —
likely the RNNT decoder's own symbol-by-symbol context needs, not chunk size, so
this lever doesn't touch the real bottleneck; (2) smaller chunks mean far more
forward passes per second of audio, so RTF rises sharply (0.02→0.18) for a latency
return that's within noise. Net: at every chunk size tested, Nemotron's real
first-word latency (TTFT ~2.2-3.3s) stays worse than or comparable to Whisper's
(TTFT p50 2.0s at 0.6s segments, with far better WER at every comparable point),
and the pretrained default `[70,13]` is arguably already the best tradeoff on this
axis, not something to move away from.

Hyp TSVs: `nemotron_fullft_3way_{legal_seed3,merged_seed1}/streaming_hyps_ac{70_13,70_6,70_1,70_0}.tsv`.

---

## Three-domain, three-seed Whisper re-run on an audited split (2026-07-13)

Follow-up request: repeat the Whisper streaming experiment against a new,
independently-audited three-way split
(`/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1`) covering **all
three domains** — legal, legacy, merged — with **3 training seeds each** (9 finetune
runs), so both offline and streaming numbers are a real range instead of a single
draw. `split_audit.json` documents the split algorithm and provenance (seed=7,
per-domain train/val/test with sha256-logged source files) — legal 238/30/30,
legacy 76/10/10, merged 314/40/40 (ordered legal+legacy union). **Legacy had never
had an honest train/val/test split in this project before** — only legal and merged
did; every prior legacy number was informal.

### Two real bugs hit along the way

1. **`sacct`'s stated failure reason was a red herring.** Submitted all 9 finetune
   jobs under `torch_pr_39_general` (per explicit instruction to switch off
   `torch_pr_39_tandon_advanced`, which had become congested — see
   [[hpc-slurm-account]] equivalent note, now a live/switchable preference not a
   fixed rule). 4 of 9 jobs failed within ~1 minute; `sacct` reported
   `QOSMaxGRESPerUser` as the reason, which reads like a per-user GPU-count cap —
   but `sacctmgr show qos ... maxtresperuser` came back empty for this account's QOS,
   meaning there is no such cap. The real cause (below) was a Python crash; the QOS
   reason string was stale/misleading, attached from a transient scheduling moment
   before the job actually got a node and crashed on its own. Lesson: always read
   the job's own `.out` log before trusting `sacct`'s `Reason` column on a job that
   ran for any nonzero amount of time.
2. **Real bug: `split_v1`'s label CSVs carry the wrong token vocabulary.** The crash
   was `IndexError: Out of range: piece id is out of range` from `sentencepiece`
   decoding `token_ids` in the label CSV. `split_v1/labels/{domain}/{split}.csv` has
   the same 4-column shape (`dataset_name,rel_path,frame_count,token_ids`) as the
   already-regenerated `*_spm1023.csv` files used elsewhere in this project, so it
   was assumed to already be in the pipeline's current tokenization — it wasn't.
   Checked: `spm_unigram_1023.model` has `vocab_size=1023`, but `split_v1`'s
   `token_ids` go up to ~5037 — the **old unigram5000 vocabulary**
   (`spm/unigram/unigram5000_units.txt`), exactly the raw format
   `scripts/regenerate_patient_labels.py`'s own docstring warns about ("the original
   CSVs... carry token ids produced by the offline TextTransform... NOT
   SentencePiece's own ids"). Fixed by running all 9 files
   (`legal/{train,val,test}`, `legacy/{train,val,test}`, `merged/{train,val,test}`)
   through `regenerate_patient_labels.py` into
   `/scratch/pa2753/avsr_realtime/labels/splitv1/{domain}/{split}_spm1023.csv`
   (`th3482`'s directory isn't writable by `pa2753`, so output goes to our own
   scratch, matching the project's existing `LABELS_OUT_DIR` convention) — zero rows
   skipped across all 9, and spot-checked decoded text against the known reference
   clip (`do you know what they think`) to confirm correctness before resubmitting.
   Cancelled the 5 still-pending original jobs (4 had already failed) rather than
   letting them hit the same bug, fixed the sbatch's `TRAIN_FILE`/`VAL_FILE`/
   `TEST_FILE` to point at the regenerated files, and resubmitted all 9 clean.

New standalone sbatch `whisper_splitv1_finetune.sbatch` (deliberately skips
`_patient_data.sh`'s merge-symlink logic entirely, since `split_v1` is already
pre-split per-domain — just sets `ROOT_DIR`/`TRAIN_FILE`/`VAL_FILE`/`TEST_FILE`
directly). Recipe unchanged from the project's established best config: Whisper
large-v3, full-FT (`--unfreeze-encoder-layers -1 --train-decoder`), SpecAugment on,
30 epochs, `SEED` env threading into `whisper_finetune.py`'s existing `--seed`
default (the same mechanism already built to guard against the ~24pp unseeded LoRA
variance this project has seen before).

### Results

Offline (from each finetune job's own end-of-run eval against the domain's
held-out `test.csv`) and streaming (SimulStreaming, 0.6s segments — the
established latency/WER sweet spot from the earlier sweep — job 13534223, all 9
converted checkpoints verified bit-exact first, job 13531530):

| Domain (train/val/test) | Offline seed1/2/3 | Offline mean | Streaming seed1/2/3 | Streaming mean | Streaming penalty |
| --- | --- | --- | --- | --- | --- |
| Legal (238/30/30) | 6.56% / 7.65% / 8.74% | 7.65% | 14.75% / 16.94% / 11.48% | 14.39% | +6.74pp |
| Legacy (76/10/10) | 40.00% / 17.50% / 25.00% | 27.5% | 32.50% / 22.50% / 50.00% | 35.0% | +7.5pp |
| Merged (314/40/40) | 9.87% / 7.62% / 7.17% | 8.22% | 17.49% / 17.49% / 16.14% | 17.04% | +8.82pp |

Streaming latency (TTFT/word-lag) doesn't vary meaningfully by domain — it's set by
segment length and the model's own decode behavior, not by which data it's
decoding: TTFT p50 ~1.97-2.59s, word-commit-lag p50 ~1.55-1.71s across all 9 runs,
consistent with the earlier legal/merged-only 0.6s-segment numbers
(TTFT p50 1.97-2.09s, word-lag p50 1.49-1.56s).

**Legacy's high variance is real, and gets worse under streaming, with the seed
ranking flipping entirely.** Offline range is 22.5pp (17.5-40.0%); streaming range
is **27.5pp (22.5-50.0%)** — wider, not narrower. Seed3 is the **best** offline
seed (25.0%, would look like the "winning" config if only one seed had been run)
but becomes the **worst** streaming seed (50.0%) — the exact failure mode 3-seed
evaluation exists to catch, and a concrete demonstration that offline WER doesn't
reliably predict which seed will streaming-decode best, at least on a
10-clip test set. Legal (5.46pp streaming range) and merged (1.35pp) stay
comparatively tight.

The +6.7 to +8.8pp streaming penalty is consistent across all three domains and in
the same range as the original legal-only/merged investigation earlier in this
file — streaming Whisper reliably costs high-single-digit WER points versus
offline, regardless of which data domain it's evaluated on.

New scripts: `/scratch/pa2753/whisper_splitv1_finetune.sbatch`,
`/scratch/pa2753/convert_splitv1_all.py` (+ `.sbatch`, loops all 9 checkpoints with
a hard bit-exact-or-abort assert per checkpoint — deletes the output file and
reports failure rather than silently trusting a bad conversion, learned from the
corrupted-checkpoint incident earlier in this file),
`/scratch/pa2753/whisper_streaming_eval_splitv1.sbatch`. Regenerated labels at
`/scratch/pa2753/avsr_realtime/labels/splitv1/{legal,legacy,merged}/{train,val,test}_spm1023.csv`.
Checkpoints and hyp TSVs at
`/scratch/pa2753/experiments/whisper_asr/whisper_largev3_specaug_splitv1_{domain}_seed{1,2,3}/`.

---

## 2026-07-14: CarelessWhisper/WhisperRT — first real causal-streaming numbers

The causal-LoRA route flagged throughout this file as "the only real lever below
the ~1.4s AlignAtt floor" is now implemented and measured
(`asr_baselines/carelesswhisper_streaming_eval.py` + sbatches; checkout at
`/scratch/pa2753/third_party/CarelessWhisper-streaming`, env at
`/scratch/pa2753/envs/whisper_rt`, checkpoints at
`/scratch/pa2753/carelesswhisper/ckpts/`). Zero-shot released checkpoints
(LibriSpeech-trained, NO patient finetune), greedy, 300ms chunks, legal split_v1
test (30 clips), same manifest as the seed1 SimulStreaming runs:

| Model | WER | TTFT p50 | word-lag p50 | ms/chunk | RTF |
|---|---|---|---|---|---|
| small_300 (job 13552629) | 125% | **0.63s** | **0.05s** (n=16) | 33 | 0.11 |
| large-v2_300 (job 13553941) | 136% | **0.67s** | **0.15s** (n=49) | 72 | 0.25 |
| (ref) SimulStreaming large-v3 patient-FT, 0.6s seg | 14.4% | ~1.4-1.6s | ~1.35-1.5s | grows w/ buffer | — |

Two clean findings:

1. **The latency floor is genuinely broken.** TTFT ~0.65s, word commit lag
   0.05-0.15s, flat O(1) per-chunk compute. This is the sub-1s live-captioning
   regime that no amount of SimulStreaming/Nemotron parameter search reached,
   and it holds across model sizes. The architecture does what the paper
   (arXiv 2508.12301) claims.
2. **Zero-shot WER is unusable on dysarthric speech and model size does not
   help** (large-v2 is *worse* than small — both pure hallucination, e.g.
   "i did not get your point" → "I jingled a chitchat, yow, which"). The gap
   is domain, not capacity, so the WER question is entirely deferred to the
   patient LoRA finetune (pipeline built and committed:
   `make_carelesswhisper_dataset.py` → MFA align → `carelesswhisper_finetune.sbatch`;
   not yet run). Caveats for that comparison: their sizes stop at large-v2 (no
   large-v3), and their original code is CC BY-NC 4.0 (non-commercial).

Word-lag counts only words whose timed-token alignment exactly matches the
final hyp (16/49 words across 30 clips at these garbage WERs) — treat the lag
numbers as latency-mechanics evidence, not statistics, until a finetuned model
produces real transcripts.

Cluster traps hit and documented in the sbatch header: venv → conda (compute
nodes resolve `/usr/bin/python3` to 3.9, login nodes 3.12 — venv symlinks
break), pip user-site shadowing (`~/.local` py3.10 packages mask missing env
deps under the job's `PYTHONNOUSERSITE=1`), login-node `/tmp` too small for the
torch wheel, `~/.conda/pkgs` blowing the home quota.

## 2026-07-14 (same day, later): first CarelessWhisper patient finetune

Ran the finetune pipeline end-to-end for the first time: MFA-align the legal
train/val/test splits (`slurm/carelesswhisper_mfa_align.sbatch`), LoRA-finetune
`small_300` on the 238-clip legal train set
(`slurm/carelesswhisper_finetune.sbatch`), convert the Lightning checkpoint to
their eval format with a bit-exact assert
(`asr_baselines/convert_carelesswhisper_ckpt.py`), re-eval on the 30-clip
legal test set.

**MFA alignment: zero drops.** 238/238 train, 30/30 val, 30/30 test TextGrids
produced -- dysarthric legal speech aligned cleanly, no fallback to a
smaller/cleaner subset needed.

**Training: real overfitting on a small train set, as expected.** Val WER
dropped fast (epoch0 62.5% -> epoch4 35.4%) then plateaued/inverted (train
loss 3.6 -> 0.04 by epoch9 while val WER stayed flat ~35.5-36%) -- 238 clips,
10 epochs, `no_logger` cannot be passed (see fix below) so metrics came from
Lightning's own progress bar. Best checkpoint by `val/wer_epoch`: epoch 4.

**Test-set result (30 held-out clips, greedy, 300ms chunks):**

| | Zero-shot small_300 | **Finetuned small_300 (epoch 4)** | SimulStreaming large-v3 patient-FT (ref) |
|---|---|---|---|
| WER | 125% | **68.3%** | 14.4% |
| TTFT p50 | 0.63s | **0.62s** | ~1.4-1.6s |
| word-lag p50 | 0.05s | **0.05s** | ~1.35-1.5s |
| ms/chunk | 33 | **24** | grows w/ buffer |

Finetuning nearly halved the error rate (125% -> 68.3%) while the latency
floor held exactly -- the causal-streaming mechanics don't degrade under
finetuning, only the WER moves. Still far from the 14.4% SimulStreaming
reference: only 238 training clips (vs the much larger pretraining data
CarelessWhisper's own results were built on), `small` size (not large-v3, no
large-v3 option exists upstream), and only 10 epochs on a LoRA that has
clearly started overfitting by epoch 9. More data (legacy domain, more
epochs with early stopping, or a larger base model) is the next lever, not
architecture.

**New bugs found running the real pipeline (all fixed, see commits
`26ee911`..`192351e`):**
1. `train.py` has no pyaudio stub (unlike our eval script) -- crashes on
   import. Fixed: drop an equivalent stub into the env's site-packages from
   the finetune sbatch.
2. `lmdb` imported unconditionally by `datasets_classes.py` even though
   `--lmdb` isn't passed, and isn't in upstream's own requirements.txt. Added
   to ours.
3. **CSV delimiter mismatch**: `AlignedTextGridDatasetLMDB` hardcodes
   `separator='\t'` with no CLI override; our dataset builder wrote
   comma-separated CSVs, which pandas parsed into one column and
   KeyError'd deep inside DataLoader workers. Fixed: write tab-separated.
4. `whisper_rt.audio.load_audio`'s ffmpeg fallback is hit unconditionally in
   the training dataset's `__getitem__` (unlike eval, which reads via
   soundfile first) -- `ffmpeg` wasn't on PATH since the sbatch invokes the
   env's python binary directly rather than activating the env. Fixed:
   installed conda-forge ffmpeg into the env, prepended its bin/ to PATH.
5. `--no_logger` sets `Trainer(logger=False)`, but `train_model()`
   unconditionally attaches a `LearningRateMonitor` callback that requires a
   real logger -- `MisconfigurationException`. Fixed: don't pass
   `--no_logger`; force their hardcoded wandb logger offline
   (`WANDB_MODE=offline`, no account/network needed) instead.

Also hit and fixed two cluster-account/scheduling issues unrelated to
CarelessWhisper itself: MFA's `conda create -n mfa` (name-based) defaulted to
`~/.conda/envs`, re-hitting the home-quota trap already known for pip/torch --
fixed the same way (`-p` prefix on scratch). And compute nodes don't have
`conda` on PATH by default even after `module load anaconda3` on the login
node -- the sbatch now sources `conda.sh` itself.

## 2026-07-14 (same day, later still): merged domain -- does more data help?

Repeated the finetune on the merged split (314/40/40, legal+legacy union,
314 train clips vs legal-alone's 238) to test whether the WER gap is
data-limited. MFA alignment: zero drops again (314/314, 40/40, 40/40).

**Test-set result (40 held-out clips, greedy, 300ms chunks):**

| | Legal-only finetune (238 train) | **Merged finetune (314 train)** |
|---|---|---|
| Val WER (best epoch) | 35.4% (epoch 4) | **30.5%** (epoch 8) |
| Test WER | 68.3% | **56.95%** |
| TTFT p50 | 0.62s | 0.63s |
| word-lag p50 | 0.05s | 0.06s |

More data clearly helps -- test WER dropped another ~11pp with ~32% more
training clips, and best-epoch val WER improved too (35.4% -> 30.5%),
without the earlier plateau/overfit signature showing up as badly by epoch 8
(still 10 epochs, LoRA rank 32, same recipe otherwise). Latency floor is
completely unaffected by domain or checkpoint, as expected -- it's a
property of the chunk size and architecture, not the weights.

**Caveat, flagged per [[patient-avsr-wer-variance]]**: single unseeded runs
on ~230-320 train / ~30-40 test clips can swing double-digit points from
sampling alone (the AV Emformer's own LoRA runs varied ~24pp across seeds on
comparable data sizes). The legal-only -> merged improvement (68.3% ->
56.95%) is consistent with "more data helps" but is one run each, not a
seeded comparison -- treat the direction as a real signal, the exact
magnitude as noisy until multiple seeds are run.

Hit one more infra bug getting here: submitting 3 concurrent MFA `--clean`
align jobs (train/val/test) raced on the *acoustic model's* extraction
directory (not the corpus workspace -- `--temporary_directory` doesn't cover
it) and killed 2 of 3 jobs with `OSError: Stale file handle`. Real fix:
`MFA_ROOT_DIR` env var (read at process start, unlike the CLI flag) pointed
at a persistent scratch dir with models pre-downloaded once; don't run
`--clean` concurrently against a shared root regardless (see commits
`5a59748`, `e5eebd4`).

## 2026-07-14 (same day, later still again): large-v2 -- bigger is NOT better here

Tried the obvious next lever -- their biggest available size (`large-v2`,
1.6B params, no v3 exists upstream) on the merged split, same recipe as
`small` (LR 1e-5, LoRA rank 32, batch 16, 10 epochs, 300ms chunks).

**Result: large-v2 is worse than small on every axis, not better.**

| | small_300, merged (314 train) | **large-v2_300, merged (314 train)** |
|---|---|---|
| Val WER (best epoch) | 30.5% (epoch 8) | **71.9%** (epoch 3) |
| Test WER | 56.95% | **101.35%** (worse than random deletion) |
| TTFT p50 | 0.63s | 0.68s |
| word-lag p50 | 0.06s | 0.20s |
| ms/chunk | 22 | 69 |

Train loss dropped fine (down to ~0.05-0.5 by epoch 9, same shape as the
small runs) while val WER got *worse* after epoch 3 and never recovered --
this isn't underfitting, it's the LoRA/optimizer recipe not transferring to
the bigger base. Notably, their own README uses a **different** large-v2
recipe (`batch_size=4, rank=4`) from the small/base recipe (`batch_size=16,
rank=32`) we've been using throughout -- we ran large-v2 with the small
recipe's hyperparameters, and this result suggests that substitution isn't
free. Latency also got measurably worse (word-lag p50 3x higher, TTFT up),
consistent with more per-chunk compute on the same chunk size.

**Conclusion: don't retry `large-v2` with the small-recipe hyperparameters
expecting it to help.** If model-size is revisited, use their own large-v2
recipe (lower rank, smaller batch, possibly the `--random_masking` RCS
variant they used for it) rather than reusing the small recipe unchanged.
For now, the best-known configuration remains **small_300, merged-domain
finetune, 56.95% test WER** -- more data (this session's other lever)
helped; a bigger base model, transplanted naively, did not.

Also hit a new infra issue: converting the large-v2 Lightning checkpoint
(fuller optimizer state than `small`) on the login node got silently
SIGKILL'd (exit 137, no error text -- looks like a hang unless you check the
real exit code past a pipe). Fixed by running the converter as a small CPU
sbatch job instead of interactively (see commit `0aa4e28`).

## 2026-07-14 (same day, later still again x2): seed variance + full finetune

Two follow-ups on the merged `small_300` result: (1) run 3 more seeds with
`--early_stop` (their EarlyStopping callback, patience=2 on val/wer) and a
40-epoch ceiling, to get an honest sense of variance instead of one lucky/
unlucky draw; (2) try **full finetuning** instead of LoRA -- their codebase
has no CLI flag for this (`LoRAStreamedWhisper.__init__` unconditionally
freezes every non-LoRA param), so this required patching
`training_code/whisper_module.py` at runtime via an env-var-gated
conditional (`carelesswhisper_finetune.sbatch`'s `FULL_FINETUNE=1`, idempotent
patch, see commit for details). Verified `configure_optimizers` doesn't
hardcode LoRA-only param filtering (it groups by `requires_grad`, generic)
before trusting this.

**LoRA multi-seed spread (small_300, merged, early-stop, up to 40 epochs):**

| Seed | Val WER (best epoch) | Test WER |
|---|---|---|
| 3407 (original run, no early-stop, 10 fixed epochs) | 30.5% | 56.95% |
| 1 | 32.2% | 63.68% |
| 2 | 30.6% | 66.82% |
| 3 | 29.6% | 59.19% |

Range 56.95-66.82% (~10pp spread across 4 seed values), mean ~61.7%. Real
variance, confirming [[patient-avsr-wer-variance]]'s warning applies here
too -- though notably tighter than the ~24pp swings seen on the AV Emformer,
and the original 56.95% number was on the lucky end, not representative of
the mean.

**Full finetune result (small_300, merged, seed 3407, early-stop): test WER
40.81%.** Beats every single LoRA seed above by 16-26pp -- not within the
LoRA seed-variance band, a categorically different result. Latency floor
unaffected as always (TTFT p50 0.64s, word-lag p50 0.08s). Training itself
converged much faster than LoRA (loss dropped to ~0.01-0.05 within 2 epochs,
early-stopped at epoch 4 total, ~4.5 min wall-clock) -- `small`'s full
240M-param capacity fits patient speech far better than LoRA's 7.1M
trainable params allow, without the catastrophic overfitting that hurt
large-v2 (whose base is 6.5x bigger, on the same ~300-clip budget).

**Practical implication**: full finetuning `small`, not model size or LoRA
rank, is now the best-known lever for closing the gap to the 14.4%
SimulStreaming reference. 40.81% is still short of that, but it's the first
result in this whole investigation that meaningfully moves the number rather
than plateauing in the high-50s/60s. Full-FT `large-v2` was explicitly NOT
tried (see [[carelesswhisper-status]] reasoning: large-v2's LoRA result
already showed the bigger base overfitting on this data size, and full FT
gives it even more capacity to do that with, not less).

New infra note: converting a full-finetune checkpoint (not just large-v2)
also OOM'd the login node -- it's the optimizer state size that matters, not
raw base-model size (`small` full-FT has 240M trainable vs large-v2 LoRA's
7.1M, and both needed the CPU-sbatch conversion route). Also fixed a latent
concurrency bug in the finetune sbatch before running seeds in parallel:
the original SEED-patch approach sed-substituted the literal `SEED = 3407`
line in the shared checkout, which would race if multiple seed jobs started
near-simultaneously (job A patches to SEED=1, job B's sed pattern no longer
matches the now-changed line and silently no-ops). Fixed by patching the
line ONCE to read from an env var at runtime instead of literal-substituting
per run -- safe under concurrency since the file is written only once ever.
Also had to make `RUN_NAME` include the seed by default, since the upstream
checkpoint dirpath doesn't include seed and would otherwise let concurrent
seed runs clobber each other's checkpoint files.

## 2026-07-14 (same day, later still again x3): full-FT large-v2 -- the earlier verdict was wrong

Full-FT small beat every LoRA config, so the obvious next question: does
full-FT fix large-v2 too? Earlier in this file, large-v2's LoRA failure
(69.9% val WER even with their own recipe) was read as "genuinely
data-limited, not a hyperparameter artifact," and the recommendation was
explicitly **not** to try full-FT there ("more capacity to overfit 314
clips, not less"). That reasoning was wrong.

**Full-FT large-v2 (merged, batch=4 for memory headroom, early-stop):
test WER 39.91%** -- essentially tied with (marginally better than) full-FT
small's 40.81%, and 61pp better than large-v2's own LoRA result (101.35%).
Val WER: 46.8% -> 35.6% -> **26.9%** (epoch 2, best) -> 33.1% -> 33.1%
(early-stopped). No OOM at batch=4 despite unfreezing all 1.6B params
(AdamW state alone is ~2x params in extra memory).

**The real lesson, corrected**: large-v2's LoRA failure wasn't data
scarcity — it was LoRA's own trainable-parameter budget. LoRA gives a
model exactly as many knobs as its rank allows (31M for large-v2's default
rank=32, or ~8M at their own rank=4 recipe) regardless of how many total
parameters the base model has. A bigger base model needs proportionally
more trainable capacity to adapt via LoRA than a smaller one does on the
same tiny dataset -- rank=32 that's "enough" LoRA capacity for `small`
(240M base) is nowhere near enough for large-v2 (1.6B base, ~6.5x bigger).
Full finetuning sidesteps this entirely by giving every model exactly as
much capacity as it already has. Model size was never the actual variable
under test in the earlier LoRA comparison -- LoRA rank relative to base
size was.

**Updated final ranking (merged domain, all test WER on the same 40-clip
held-out set):**

| Config | Test WER | TTFT p50 | word-lag p50 |
|---|---|---|---|
| Zero-shot (either size) | 125-136% | ~0.65s | ~0.10s |
| LoRA, small (4 seeds) | 56.95-66.82% | ~0.63s | ~0.06s |
| LoRA, large-v2 (either recipe) | 101.35% | ~0.67s | ~0.15-0.20s |
| Full-FT, small (seed 3407) | 40.81% | 0.64s | 0.08s |
| **Full-FT, large-v2 (seed 3407)** | **39.91%** | 0.66s | 0.11s |
| SimulStreaming reference | 14.4% | ~1.5s | ~1.4s |

Full-FT closes roughly two-thirds of the gap between zero-shot garbage and
the SimulStreaming reference, on either base size. Latency stays well under
1s either way -- large-v2 full-FT's word-lag (0.11s) is still ~12x lower
than SimulStreaming's (1.4s), so the size choice going forward is about
compute budget and further headroom, not latency or (apparently) achievable
WER -- both sizes seem to land in the same ballpark once given real
capacity to adapt. Only one seed tried per full-FT config so far; per
[[patient-avsr-wer-variance]], treat the small-vs-large-v2 tie as
directional, not confirmed, until seeded.

## 2026-07-16: SpecAugment closes another chunk of the gap

Their codebase has zero data augmentation. Full-FT's fast convergence with
train loss near zero by epoch 4-6 (both small and large-v2 full-FT runs)
was a clear memorization signature on ~314 training clips -- a natural next
lever. Wired in this repo's existing Green-parameterized SpecAugment
(`asr_baselines/specaugment.py` -- cut frequency masking, blown-up time
masking, already used for the Whisper/Nemotron pipelines) via a new
`SPECAUG=1` flag on `carelesswhisper_finetune.sbatch`. Their dataset classes
have no hook for this, so it required patching `datasets_classes.py`
(`AlignedTextGridDatasetLMDB.__getitem__`, applying the mask to the computed
mel spectrogram on the train split only) and `whisper_module.py`'s
`get_dataset` (which never passed a real `augment`/`split` distinction
through, despite a vestigial unused `split` constructor param already
existing) -- both patches idempotent, env-var-gated, dry-run-verified
against copies of the real cluster files (not just locally compiled) before
deploying, same pattern as the other vendored-code patches in this file.

**Full-FT + SpecAugment, small, merged, seed 3407: test WER 34.08%** --
down from 40.81% without augmentation. Val WER also improved (22.4% -> 20.0%
best epoch), consistent with augmentation genuinely reducing overfitting
rather than just adding eval-time noise. Latency floor unaffected as always
(TTFT p50 0.62s, word-lag p50 0.04s).

**Updated best-known result: 34.08% test WER** (full-FT + SpecAugment,
small, merged domain) -- the strongest number in the whole investigation,
closing roughly three-quarters of the gap between zero-shot garbage (125%)
and the 14.4% SimulStreaming reference. Not yet tried: SpecAugment on
large-v2 full-FT (should compose similarly given the mechanism is
architecture-independent), or combined with more data/longer training.
Still one seed only.

## 2026-07-16 (later): SpecAugment on large-v2 -- mixed, not a clean win

Tried the obvious follow-up: SpecAugment + full-FT on large-v2 (same
`BATCH_SIZE=4` memory-safety setting as before). First attempt (with
`--early_stop`) was a clear failure: val WER stuck at 72-73% across 3
epochs, early-stopped almost immediately, test WER **92.38%** -- worse than
doing nothing. Train loss and the pattern (val WER *worsening* epoch 0->1)
made this look like the run was cut off before real adaptation started, not
a genuine plateau -- their `patience=2` was tuned against the unaugmented
large-v2 full-FT run's convergence speed (reached 26.9% val WER by epoch 2),
and augmentation plus the small `batch=4` (noisier per-step gradients, no
CLI patience override exists) needed more epochs before showing signal.

Reran without `--early_stop`, fixed 15 epochs: val WER dropped steadily
(51.7% -> 38.2% -> 27.7% ... -> **20.8%** at epoch 13, still trending down
noisily, not clearly plateaued) -- confirms the early-stop theory. But the
**test WER on the best checkpoint (epoch 13): 41.70%** -- worse than
large-v2 full-FT WITHOUT SpecAugment (39.91%), despite the much better val
number. Latency floor still fine (TTFT p50 0.69s, word-lag p50 0.11s).

**Honest reading: SpecAugment does NOT reproduce its `small`-model win on
large-v2.** Small: 40.81% -> 34.08% (clear win). Large-v2: 39.91% -> 41.70%
(marginally worse, within likely seed/eval noise on a 40-clip test set, but
not an improvement either way). Plausible explanations, untested: large-v2's
much smaller `batch=4` interacts with SpecAugment's default masking
aggressiveness differently than `small`'s `batch=16` (proportionally more
of each batch masked-in-effect at small batch size); or the val-epoch-13
checkpoint, while best on val, is a val/test mismatch artifact on this tiny
test set; or large-v2 genuinely needs a lighter augmentation config than the
Green defaults tuned for the smaller model. Not chased further this
session -- **the practical best-known config remains full-FT + SpecAugment
on `small` (34.08%)**, which is also ~6x cheaper to train.

Also worth noting for reproducibility: two runs of nominally the "same"
epoch 0 (same seed, same config, only `--early_stop` differed) produced
very different val WER (72.0% vs 51.7%) -- likely DataLoader worker
non-determinism (16 workers requested on an 8-core node, a warning already
present in every run's log) rather than a real seed effect. Single-epoch
comparisons on this pipeline are not reliable; only trust multi-epoch
trends and final test numbers.

---

## Environmental hazards to know about

- **`/home/pa2753` is at/near its inode quota** on the torch cluster — a `touch` failed even after freeing ~180 files. Not caused by this investigation specifically (pre-existing), but will block any future work that writes many small files there. Established mitigation pattern this whole project: keep envs, caches, and any file-heavy third-party code on `/scratch/pa2753/` instead of `/home/pa2753/`.
- SSH to the cluster (`ssh torch`) needs periodic manual re-authentication when the control-master socket expires — shows up as `Permission denied (gssapi-keyex,...)` and needs the user to run `ssh torch` interactively once to restore it. Also occasionally needs the NYU VPN reconnected (internal `10.x.x.x` addresses aren't reachable without it — distinguishable from a real outage by testing a generic external host like `github.com`, which will succeed while `torch` still fails).
