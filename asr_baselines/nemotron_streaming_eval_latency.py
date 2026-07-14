#!/usr/bin/env python3
"""Cache-aware streaming eval for Nemotron with TTFT/word-lag (simulated real-time).

Extends nemotron_streaming_eval.py with the same latency metrics as
whisper_streaming_eval.py: TTFT (stream start -> first text visible) and
word-commit lag (word finished being spoken -> its text visible), both in
SIMULATED REAL TIME (chunk k's audio only exists at (k+1)*chunk_shift seconds
in a live deployment). Uses streaming_buffer.buffer_idx / streams_length
(fraction of this utterance's feature frames consumed) times the manifest
duration to get audio_avail seconds, so it needs no hardcoded frame-stride
constant and works across any streaming_cfg.

Word timestamps: DISABLED. compute_timestamps=True is incompatible with
cache-aware streaming's carried partial-hypothesis state on BOTH of NeMo's
greedy RNNT decode paths (confirmed independently on nemo_toolkit==2.7.3):
loop_labels=True crashes in hyp.merge_() (hyp.timestamp is a dict, .extend()
called on it -- AttributeError); loop_labels=False raises
_greedy_decode_blank_as_pad_loop_frames's own
NotImplementedError("`partial_hypotheses` support is not supported"). TTFT
(time-to-first-text) is still measured; per-word commit lag is not available
until NeMo fixes this.

  python -m asr_baselines.nemotron_streaming_eval_latency \
    --model $EXP/nemotron_patient.nemo --manifest test.jsonl \
    --att-context-size 70,6 --out streaming_hyps.tsv
"""

import argparse
import json
import os
import statistics
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.metrics import compute_wer_cer  # noqa: E402


def read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--chunk-size", type=int, default=None,
                    help="raw chunk_size override -- requires --shift-size too (rarely "
                         "what you want; prefer --att-context-size)")
    p.add_argument("--shift-size", type=int, default=None)
    p.add_argument("--left-chunks", type=int, default=None)
    p.add_argument("--att-context-size", type=str, default=None,
                    help="comma-separated pair 'left,right' selecting one of the "
                         "model's discrete streaming lookahead presets (e.g. "
                         "nemotron-speech-streaming-en-0.6b supports 70,13 (default, "
                         "~1.12s/chunk) / 70,6 (~0.56s) / 70,1 (~0.16s) / 70,0 (~0.08s, "
                         "fully causal) -- smaller right value = lower latency, higher WER)")
    p.add_argument("--online-normalization", action="store_true")
    args = p.parse_args()

    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model.endswith(".nemo") and os.path.isfile(args.model):
        model = nemo_asr.models.ASRModel.restore_from(args.model, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location=device)
    model.eval()
    model = model.to(device)

    if args.att_context_size is not None:
        left, right = (int(x) for x in args.att_context_size.split(","))
        model.encoder.setup_streaming_params(att_context_size=[left, right], left_chunks=args.left_chunks)
    elif args.chunk_size is not None or args.left_chunks is not None:
        model.encoder.setup_streaming_params(
            chunk_size=args.chunk_size, shift_size=args.shift_size, left_chunks=args.left_chunks)
    cfg = model.encoder.streaming_cfg
    print(f"streaming_cfg: chunk_size={cfg.chunk_size} shift_size={cfg.shift_size} "
          f"pre_encode_cache_size={cfg.pre_encode_cache_size}")

    items = read_manifest(args.manifest)
    refs, hyps, rows = [], [], []
    total_audio = 0.0
    total_elapsed = 0.0
    total_chunks = 0
    ttft_sims = []
    dev = next(model.parameters()).device

    for it in items:
        path = it["audio_filepath"]
        dur = it.get("duration", 0.0)
        total_audio += dur

        streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=args.online_normalization)
        streaming_buffer.append_audio_file(path, stream_id=-1)
        total_feat_frames = float(streaming_buffer.streams_length[0].item())

        cache_last_channel, cache_last_time, cache_last_channel_len = model.encoder.get_initial_cache_state(batch_size=1)
        previous_hypotheses = None
        pred_out_stream = None
        transcribed_texts = None
        prev_text_len = 0
        utt_ttft = None

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
                    processed_signal=chunk_audio.to(dev),
                    processed_signal_length=chunk_lengths.to(dev),
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

            frac = min(streaming_buffer.buffer_idx, total_feat_frames) / total_feat_frames
            audio_avail = frac * dur
            emit_sim = audio_avail + compute

            hyp_obj = transcribed_texts[0]
            hyp_text = hyp_obj.text if hasattr(hyp_obj, "text") else hyp_obj
            if hyp_text and len(hyp_text) > prev_text_len:
                if utt_ttft is None:
                    utt_ttft = emit_sim
                    ttft_sims.append(emit_sim)
                prev_text_len = len(hyp_text)
        total_elapsed += time.time() - t0

        hyp = transcribed_texts[0]
        hyp = hyp.text if hasattr(hyp, "text") else hyp
        refs.append(it["text"])
        hyps.append(hyp)
        rows.append((path, it["text"], hyp))

    m = compute_wer_cer(refs, hyps)
    rtf = total_elapsed / total_audio if total_audio else float("nan")
    print(f"N={len(refs)}  streaming WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"RTF={rtf:.3f}  chunks={total_chunks}  "
          f"{1000*total_elapsed/max(total_chunks,1):.1f} ms/chunk  "
          f"({total_elapsed:.1f}s / {total_audio:.1f}s audio)")

    def pctl(xs, q):
        return sorted(xs)[min(len(xs) - 1, int(q * (len(xs) - 1)))]
    if ttft_sims:
        print(f"TTFT (stream start -> first text, simulated real-time): "
              f"mean {statistics.mean(ttft_sims):.2f}s  p50 {pctl(ttft_sims,.5):.2f}s  "
              f"p95 {pctl(ttft_sims,.95):.2f}s  (n={len(ttft_sims)})")
    print("word commit lag: unavailable (compute_timestamps incompatible with "
          "cache-aware streaming's partial-hypothesis carry in this NeMo version)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\n")
            for path, ref, hyp in rows:
                f.write(f"{path}\t{ref}\t{hyp}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
