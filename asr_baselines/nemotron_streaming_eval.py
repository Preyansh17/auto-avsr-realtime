#!/usr/bin/env python3
"""Cache-aware streaming eval for Nemotron (chunked decode, carried state).

Follow-up to nemotron_eval.py's offline transcribe(). Nemotron
(nvidia/nemotron-speech-streaming-en-0.6b) is a cache-aware FastConformer-RNNT
-- built to run in fixed-size audio chunks with encoder state
(cache_last_channel/cache_last_time) carried across chunks, the same
streaming contract as the AV Emformer. Offline transcribe() decodes the whole
utterance at once and reports a WER ceiling that chunked/causal inference may
not actually reach -- this script reports the real streaming-mode WER, plus
per-chunk latency (ms/chunk), the number actually comparable to the AV
Emformer's own streaming latency.

Uses NeMo's own cache-aware streaming utilities (same API as NeMo's
examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py
and the "Cache Aware Streaming" tutorial notebook) rather than reimplementing
chunk framing / cache bookkeeping: CacheAwareStreamingAudioBuffer for chunk
framing + model.conformer_stream_step() for the carried-state decode loop.
This API is version sensitive (like everything else in the nemotron phase --
see requirements-nemotron.txt); it has NOT been smoke-tested against the
installed nemo_toolkit==2.7.3 yet (no local NeMo install to verify against --
must be smoke-tested on a real compute-node job before trusting the numbers).

  python -m asr_baselines.nemotron_streaming_eval \
    --model $EXP/nemotron_patient.nemo --manifest val.jsonl \
    --out streaming_hyps.tsv

--word-lag adds the Nemotron counterpart to whisper_streaming_eval.py's TTFT /
word-commit-lag metrics, via NeMo's RNNT per-word frame-offset timestamps
(model.cfg.decoding.compute_timestamps=True). Frame-synchronous by
construction (each word's offset is the encoder frame it was decoded from),
so in principle more precise than Whisper's attention-derived word timestamps
-- but the exact offset semantics were confirmed only by empirical probing on
nemo_toolkit==2.7.3 (no official docs read for this), and required a
workaround for a real library bug: compute_timestamps=True turns
Hypothesis.timestamp from a list into a dict, but conformer_stream_step's
carried-state merge on the following chunk assumes it's still a list and
crashes (AttributeError: 'dict' object has no attribute 'extend') --
worked around by restoring the list form after extracting each step's new
words, before the object re-enters the merge path on the next chunk.
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.metrics import compute_wer_cer, normalize  # noqa: E402
from asr_baselines.nemotron_restore import restore_asr_model  # noqa: E402
from asr_baselines.mfa_timings import (  # noqa: E402
    build_textgrid_index, ref_word_ends, match_hyp_to_ref,
)


def read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--chunk-size", type=int, default=None,
                   help="encoder chunk size in frames; default keeps the model's "
                        "pretrained streaming_cfg (nemotron ships tuned for low-latency "
                        "streaming already)")
    p.add_argument("--left-chunks", type=int, default=None,
                   help="left-context chunks kept in cache; default keeps pretrained config")
    p.add_argument("--att-context-size", type=str, default=None,
                   help="comma-separated pair 'left,right' selecting one of the model's "
                        "discrete streaming lookahead presets (e.g. "
                        "nemotron-speech-streaming-en-0.6b supports 70,13 (default, "
                        "~1.12s/chunk) / 70,6 (~0.56s) / 70,1 (~0.16s) / 70,0 (~0.08s, "
                        "fully causal) -- the correct lever, unlike --chunk-size/"
                        "--left-chunks above which require both to be set and rarely do "
                        "what you want")
    p.add_argument("--online-normalization", action="store_true",
                   help="use running mean/var feature normalization instead of "
                        "per-utterance stats -- offline transcribe() normalizes over the "
                        "FULL utterance, which peeks at future audio a true streaming "
                        "deployment can't see; this flag removes that unfairness")
    p.add_argument("--mfa-aligned-dir", default=None,
                   help="directory of Montreal Forced Aligner .TextGrid files. "
                        "When set (with --word-lag), latency is ALSO reported "
                        "against MFA ground-truth word-end times instead of the "
                        "model's own predicted timestamps. RNN-T emission is "
                        "delayed by construction, so the model's own timestamps "
                        "are late and -- being the subtrahend -- make the "
                        "measured lag look SMALLER than it is. See "
                        "asr_baselines/mfa_timings.py.")
    p.add_argument("--mfa-split-root", default="/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1",
                   help="split root stripped from audio paths to derive the "
                        "flattened MFA utterance name")
    p.add_argument("--word-lag", action="store_true",
                   help="report per-word commit lag (word spoken -> text visible) and "
                        "TTFT, the Nemotron counterpart to whisper_streaming_eval.py's "
                        "same metrics. Opt-in: leaves default WER-only runs byte-identical "
                        "to before this flag existed.")
    args = p.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from omegaconf import open_dict

    # Handles adapter/LoRA checkpoints, which NeMo's plain restore_from()
    # cannot load -- see asr_baselines/nemotron_restore.py for the root cause.
    model = restore_asr_model(args.model)
    model.eval()

    if args.att_context_size is not None:
        left, right = (int(x) for x in args.att_context_size.split(","))
        model.encoder.setup_streaming_params(att_context_size=[left, right], left_chunks=args.left_chunks)
    elif args.chunk_size is not None or args.left_chunks is not None:
        model.encoder.setup_streaming_params(chunk_size=args.chunk_size, left_chunks=args.left_chunks)
    cfg = model.encoder.streaming_cfg
    print(f"streaming_cfg: chunk_size={cfg.chunk_size} shift_size={cfg.shift_size} "
          f"pre_encode_cache_size={cfg.pre_encode_cache_size}")

    if args.word_lag:
        # Word-level timestamps: NeMo's RNNT decoding attaches frame-index
        # offsets per word (verified empirically on this nemo_toolkit==2.7.3 --
        # not documented as stable across versions). Purely additive metadata
        # over already-decided greedy tokens; does not change WER.
        with open_dict(model.cfg):
            model.cfg.decoding.compute_timestamps = True
            model.cfg.decoding.greedy.compute_timestamps = True
        model.change_decoding_strategy(model.cfg.decoding)
        subsampling_factor = model.encoder.subsampling_factor
        window_stride = model.cfg.preprocessor.window_stride
        frame_to_sec = subsampling_factor * window_stride
        print(f"word-lag: subsampling_factor={subsampling_factor} "
              f"window_stride={window_stride} -> frame_to_sec={frame_to_sec}")

    items = read_manifest(args.manifest)
    refs, hyps = [], []
    total_audio = 0.0
    total_elapsed = 0.0
    total_chunks = 0
    device = next(model.parameters()).device
    ttft_sims = []             # stream start -> first text visible (includes leading silence)
    ttft_from_speech_sims = []  # first word's own end-of-speech -> its text visible
    word_lags = []             # word finished being SPOKEN -> its text visible
    mfa_word_lags = []         # same, but anchored to MFA ground truth
    mfa_ttft_sims = []
    mfa_index = build_textgrid_index(args.mfa_aligned_dir) if args.mfa_aligned_dir else None
    mfa_covered = mfa_missing = 0
    if mfa_index is not None:
        print(f"MFA reference: {len(mfa_index)} TextGrids indexed from {args.mfa_aligned_dir}")

    for it in items:
        path = it["audio_filepath"]
        total_audio += it.get("duration", 0.0)

        # One utterance == one stream: fresh buffer + cache per file. Matches
        # a real single-stream deployment (one live audio feed), and keeps
        # per-utterance timing directly comparable to the AV Emformer's own
        # per-utterance streaming latency.
        streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=args.online_normalization)
        streaming_buffer.append_audio_file(path, stream_id=-1)

        cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(batch_size=1)
        previous_hypotheses = None
        pred_out_stream = None
        transcribed_texts = None
        # Latency in SIMULATED REAL TIME, same convention as
        # whisper_streaming_eval.py: the eval loop feeds chunks as fast as the
        # GPU eats them, but a live user records chunk k over [k*shift, (k+1)*
        # shift], so its text can appear no earlier than that chunk's audio
        # becoming available plus this step's compute. Valid while per-chunk
        # compute < chunk duration (RTF<1).
        audio_consumed_sec = 0.0
        prev_word_count = 0
        utt_ttft = None
        emitted = []   # (word_text, emit_sim) in emission order, for MFA matching

        t0 = time.time()
        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
            tc = time.time()
            with torch.inference_mode():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio.to(device),
                    processed_signal_length=chunk_lengths.to(device),
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=cfg.drop_extra_pre_encoded if step_num != 0 else 0,
                    return_transcription=True,
                )
            compute = time.time() - tc
            total_chunks += 1
            if args.word_lag:
                # chunk_lengths is in PRE-subsampling feature-frame units (same
                # hop as window_stride, before the encoder's own subsampling_factor
                # collapses them) -- confirmed by the CacheAwareStreamingAudioBuffer
                # feeding processed_signal (mel features) rather than raw audio.
                audio_consumed_sec += chunk_lengths.item() * window_stride
                emit_sim = audio_consumed_sec + compute
                step_hyp = transcribed_texts[0]
                ts = getattr(step_hyp, "timestamp", None)
                if isinstance(ts, dict):
                    words = ts.get("word", [])
                    if words and utt_ttft is None:
                        utt_ttft = emit_sim
                        ttft_sims.append(emit_sim)
                        first_word_end_sec = float(words[0]["end_offset"]) * frame_to_sec
                        ttft_from_speech_sims.append(emit_sim - first_word_end_sec)
                    for w in words[prev_word_count:]:
                        word_end_sec = float(w["end_offset"]) * frame_to_sec
                        word_lags.append(emit_sim - word_end_sec)
                        emitted.append((str(w.get("word", "")), emit_sim))
                    prev_word_count = len(words)
                    # WORKAROUND: compute_timestamps=True turns hyp.timestamp
                    # into a dict, but conformer_stream_step's carried-state
                    # merge_() on the NEXT chunk does self.timestamp.extend(...),
                    # which requires a list. previous_hypotheses is this SAME
                    # object (aliased, not copied) fed straight back in above --
                    # restore the raw list form so the next step's merge doesn't
                    # crash with AttributeError: 'dict' object has no attribute
                    # 'extend'. Verified empirically (job-based probe) that this
                    # doesn't lose any decoding state -- only the timestamp
                    # bookkeeping gets rebuilt (from the same underlying data)
                    # every step.
                    step_hyp.timestamp = ts["timestep"]
        total_elapsed += time.time() - t0

        if mfa_index is not None and emitted:
            # Anchor to ground truth instead of the model's own (late) guess.
            # Only exact word matches are timed: a substituted or hallucinated
            # word has no reference onset to measure against.
            ref_pairs = ref_word_ends(path, mfa_index, args.mfa_split_root)
            if ref_pairs is None:
                mfa_missing += 1
            else:
                mfa_covered += 1
                hyp_norm = [normalize(w).split()[0] if normalize(w) else ""
                            for w, _ in emitted]
                ref_norm = [w for w, _ in ref_pairs]
                matches = match_hyp_to_ref(hyp_norm, ref_norm)
                for k, (hi, ri) in enumerate(matches):
                    lag = emitted[hi][1] - ref_pairs[ri][1]
                    mfa_word_lags.append(lag)
                    if k == 0:
                        mfa_ttft_sims.append(lag)

        hyp = transcribed_texts[0]
        hyp = hyp.text if hasattr(hyp, "text") else hyp
        refs.append(it["text"])
        hyps.append(hyp)

    m = compute_wer_cer(refs, hyps)
    rtf = total_elapsed / total_audio if total_audio else float("nan")
    print(f"N={len(refs)}  streaming WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"RTF={rtf:.3f}  chunks={total_chunks}  "
          f"{1000*total_elapsed/max(total_chunks,1):.1f} ms/chunk  "
          f"({total_elapsed:.1f}s / {total_audio:.1f}s audio)")

    def pctl(xs, q):
        return sorted(xs)[min(len(xs) - 1, int(q * (len(xs) - 1)))]
    if ttft_sims:
        import statistics
        print(f"TTFT (stream start -> first text, simulated real-time): "
              f"mean {statistics.mean(ttft_sims):.2f}s  p50 {pctl(ttft_sims, .5):.2f}s  "
              f"p95 {pctl(ttft_sims, .95):.2f}s  (n={len(ttft_sims)}; includes any "
              f"leading silence before the first word)")
    if ttft_from_speech_sims:
        import statistics
        print(f"TTFT from first word spoken (excludes leading silence): "
              f"mean {statistics.mean(ttft_from_speech_sims):.2f}s  "
              f"p50 {pctl(ttft_from_speech_sims, .5):.2f}s  "
              f"p95 {pctl(ttft_from_speech_sims, .95):.2f}s  (n={len(ttft_from_speech_sims)})")
    if mfa_word_lags:
        print(f"[MFA ground truth] TTFT from first word spoken: "
              f"mean {statistics.mean(mfa_ttft_sims):.2f}s  "
              f"p50 {pctl(mfa_ttft_sims, .5):.2f}s  "
              f"p95 {pctl(mfa_ttft_sims, .95):.2f}s  (n={len(mfa_ttft_sims)})")
        print(f"[MFA ground truth] word commit lag: "
              f"mean {statistics.mean(mfa_word_lags):.2f}s  "
              f"p50 {pctl(mfa_word_lags, .5):.2f}s  "
              f"p95 {pctl(mfa_word_lags, .95):.2f}s  "
              f"(n={len(mfa_word_lags)} matched words; "
              f"{mfa_covered} clips aligned, {mfa_missing} unaligned)")
    if word_lags:
        import statistics
        print(f"word commit lag (word spoken -> text visible): "
              f"mean {statistics.mean(word_lags):.2f}s  p50 {pctl(word_lags, .5):.2f}s  "
              f"p95 {pctl(word_lags, .95):.2f}s  (n={len(word_lags)} words)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\n")
            for it, ref, hyp in zip(items, refs, hyps):
                f.write(f"{it['audio_filepath']}\t{ref}\t{hyp}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
