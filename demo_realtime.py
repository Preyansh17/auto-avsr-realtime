#!/usr/bin/env python3
"""File-based real-time AVSR demo (Emformer RNN-T, streaming decode).

Streams an existing video file chunk-by-chunk through the streaming
pipeline as if it arrived live, carrying RNN-T decoder state across
chunks and printing the transcript incrementally.

Examples:
  # Pretrained device_avsr tutorial model (auto-downloads JIT + spm assets),
  # face-crop preprocessing, paced like a live source:
  python demo_realtime.py --video clip.mp4 --simulate-realtime

  # Fine-tuned eager checkpoint on an already-cropped patient mouth-ROI mp4:
  python demo_realtime.py --video roi.mp4 --checkpoint cpts/model.ckpt --preprocess roi
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.streaming import (  # noqa: E402
    RATE_RATIO,
    iter_av_windows,
    load_eager_pipeline,
    load_jit_pipeline,
    load_media_file,
)
from online_avsr.text import compute_wer  # noqa: E402

DEFAULT_SP_MODEL = os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model")


def resolve_assets(args):
    if not args.checkpoint and not args.jit_model:
        from torchaudio.utils import download_asset

        print("No model given; downloading device_avsr tutorial assets...", file=sys.stderr)
        args.jit_model = download_asset("tutorial-assets/device_avsr_model.pt")
        if not os.path.isfile(args.sp_model_path):
            args.sp_model_path = download_asset("tutorial-assets/spm_unigram_1023.model")
    if not os.path.isfile(args.sp_model_path):
        raise FileNotFoundError(
            f"SentencePiece model not found: {args.sp_model_path} (run scripts/download_assets.py)"
        )


def read_reference(value):
    if not value:
        return None
    if value.startswith("@"):
        with open(value[1:], encoding="utf-8") as f:
            return f.read().strip()
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Video file to stream")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--checkpoint", help="Eager OnlineAVSRModule checkpoint (.ckpt/.pth)")
    model.add_argument("--jit-model", help="device_avsr tutorial TorchScript archive (.pt)")
    parser.add_argument("--sp-model-path", default=DEFAULT_SP_MODEL)
    parser.add_argument(
        "--preprocess", choices=["face", "mouth", "roi", "none"], default=None,
        help="face: mediapipe face crop (tutorial JIT default); mouth: auto-avsr mouth ROI via detector; "
        "roi: input already mouth-cropped (eager default); none: raw resize, smoke tests only",
    )
    parser.add_argument("--detector", choices=["mediapipe", "retinaface"], default="mediapipe")
    parser.add_argument("--face-resize", type=int, default=None,
                        help="Face crop resize (default: 44 for JIT, 88 for eager)")
    parser.add_argument("--segment-frames", type=int, default=None,
                        help="New frames consumed per decode step (default: model segment; 32 for JIT)")
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=10)
    carry = parser.add_mutually_exclusive_group()
    carry.add_argument("--carry-state", dest="carry_state", action="store_true", default=None)
    carry.add_argument("--no-carry-state", dest="carry_state", action="store_false")
    parser.add_argument("--simulate-realtime", action="store_true",
                        help="Pace chunks at media rate; warn when decoding falls behind")
    parser.add_argument("--reference", default=None, help="Reference transcript (or @file) for WER")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    resolve_assets(args)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"

    if args.jit_model:
        preprocess = args.preprocess or "face"
        face_resize = args.face_resize or 44
        carry_state = args.carry_state if args.carry_state is not None else False
        pipeline = load_jit_pipeline(
            args.jit_model, args.sp_model_path, device=device, preprocess=preprocess,
            detector=args.detector, beam_width=args.beam_width, carry_state=carry_state,
            face_resize=face_resize,
        )
        # Tutorial cadence: context+buffer frames per call, features untrimmed;
        # the window tail acts as the Emformer right context.
        step = args.segment_frames or 32
        lookback, lookahead, trim = args.context_frames, 0, 0
        model_desc = f"jit:{os.path.basename(args.jit_model)}"
    else:
        carry_state = args.carry_state if args.carry_state is not None else True
        pipeline = load_eager_pipeline(
            args.checkpoint, args.sp_model_path, device=device,
            preprocess=args.preprocess, detector=args.detector,
            beam_width=args.beam_width, carry_state=carry_state, face_resize=args.face_resize,
        )
        backend = pipeline.backend
        preprocess = pipeline.preprocess_mode
        arch = backend.module.architecture
        step = args.segment_frames or backend.segment_length
        lookback, lookahead, trim = args.context_frames, backend.right_context_length, args.context_frames
        model_desc = (
            f"ckpt:{os.path.basename(args.checkpoint)} "
            f"(arch={arch}, segment={backend.segment_length}, rc={backend.right_context_length})"
        )

    video, audio, fps = load_media_file(args.video)
    total_frames = min(len(video), audio.size(0) // RATE_RATIO)
    media_duration = total_frames / fps
    if not 24 <= fps <= 30:
        print(f"warning: video is {fps:.1f} fps; the model expects ~25 fps (rate ratio 640)", file=sys.stderr)

    print(f"model: {model_desc}", file=sys.stderr)
    print(f"video: {args.video} ({total_frames} frames @ {fps:.1f} fps, {media_duration:.1f}s)", file=sys.stderr)
    print(
        f"streaming: {step} frames/step (+{lookback} past ctx, +{lookahead} lookahead), "
        f"preprocess={preprocess}, carry_state={carry_state}, device={device}",
        file=sys.stderr,
    )
    algorithmic_latency = (step + lookahead) / fps
    print(f"algorithmic latency: {algorithmic_latency * 1000:.0f} ms\n", file=sys.stderr)

    jsonl = open(args.output_jsonl, "a", encoding="utf-8") if args.output_jsonl else None
    chunk_walls = []
    first_token_at = None
    behind_warned = False
    stream_start = time.perf_counter()

    for start, end, v_win, a_win in iter_av_windows(
        video, audio, step_frames=step, lookback_frames=lookback, lookahead_frames=lookahead
    ):
        t0 = time.perf_counter()
        transcript, new, _ = pipeline.infer_chunk(v_win, a_win, trim_frames=trim)
        wall = time.perf_counter() - t0
        chunk_walls.append(wall)
        media_sec = (end - start) / fps
        if new.strip() and first_token_at is None:
            first_token_at = (end / fps, time.perf_counter() - stream_start)
        print(new, end="", flush=True)

        if jsonl:
            jsonl.write(json.dumps({
                "event": "partial",
                "video_path": args.video,
                "chunk_start_sec": start / fps,
                "chunk_end_sec": end / fps,
                "new_text": new,
                "partial_transcript": transcript,
                "wall_sec": wall,
                "media_sec": media_sec,
                "rtf": wall / media_sec if media_sec else None,
            }, ensure_ascii=False) + "\n")

        if args.simulate_realtime:
            media_clock = end / fps
            wall_clock = time.perf_counter() - stream_start
            if wall_clock < media_clock:
                time.sleep(media_clock - wall_clock)
            elif not behind_warned and wall_clock > media_clock + algorithmic_latency:
                print(f"\nwarning: decoder is {wall_clock - media_clock:.1f}s behind real time", file=sys.stderr)
                behind_warned = True

    total_wall = sum(chunk_walls)
    transcript = pipeline.transcript.strip()
    print("\n", file=sys.stderr)

    chunk_media = step / fps
    rtfs = [w / chunk_media for w in chunk_walls]
    stats = {
        "event": "final",
        "video_path": args.video,
        "transcript": transcript,
        "chunks": len(chunk_walls),
        "media_sec": media_duration,
        "decode_wall_sec": total_wall,
        "overall_rtf": total_wall / media_duration if media_duration else None,
        "chunk_rtf_mean": statistics.mean(rtfs) if rtfs else None,
        "chunk_rtf_p95": sorted(rtfs)[int(0.95 * (len(rtfs) - 1))] if rtfs else None,
        "algorithmic_latency_sec": algorithmic_latency,
        "time_to_first_token_sec": first_token_at[1] if first_token_at else None,
    }
    reference = read_reference(args.reference)
    if reference:
        stats["reference"] = reference
        stats["wer"] = compute_wer(reference, transcript)

    print(f"transcript: {transcript}", file=sys.stderr)
    print(
        f"chunks={stats['chunks']}  overall RTF={stats['overall_rtf']:.3f}  "
        f"chunk RTF mean={stats['chunk_rtf_mean']:.3f} p95={stats['chunk_rtf_p95']:.3f}  "
        f"algorithmic latency={algorithmic_latency * 1000:.0f}ms",
        file=sys.stderr,
    )
    if "wer" in stats:
        print(f"WER vs reference: {stats['wer']:.3f}", file=sys.stderr)
    if jsonl:
        jsonl.write(json.dumps(stats, ensure_ascii=False) + "\n")
        jsonl.close()


if __name__ == "__main__":
    main()
