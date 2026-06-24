#!/usr/bin/env python3
"""Measure end-to-end streaming AVSR latency on any video.

Breaks each streaming step into preprocess (face/mouth detect + crop + resize),
frontend (ResNet/Linear + fusion + Emformer), and decode (RNN-T beam search),
and reports:
  - inference-only RTF  (frontend + decode) / media           <- model speed
  - end-to-end RTF      (preprocess + frontend + decode) / media  <- deployable speed
  - algorithmic latency (segment + right_context) / fps          <- inherent lag
  - per-chunk wall p50/p95

Run on the HPC GPU (omit --cpu). Works on any video; pass --preprocess to match
how the model was trained (roi for pre-cropped patient clips, face for raw).

Usage:
  python scripts/benchmark_latency.py --video clip.mp4 \
    --checkpoint cpts/online_avsr_bootstrap.ckpt --preprocess roi
"""

import argparse
import os
import statistics
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from online_avsr.streaming import (  # noqa: E402
    RATE_RATIO,
    iter_av_windows,
    load_eager_pipeline,
    load_jit_pipeline,
    load_media_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--checkpoint")
    model.add_argument("--jit-model")
    parser.add_argument("--sp-model-path", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"))
    parser.add_argument("--preprocess", choices=["face", "mouth", "roi", "none"], default=None)
    parser.add_argument("--detector", choices=["mediapipe", "retinaface"], default="mediapipe")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--segment-frames", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up chunks excluded from stats")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    if not args.checkpoint and not args.jit_model:
        from torchaudio.utils import download_asset

        args.jit_model = download_asset("tutorial-assets/device_avsr_model.pt")

    if args.jit_model:
        pipeline = load_jit_pipeline(args.jit_model, args.sp_model_path, device=device,
                                     preprocess=args.preprocess or "face", detector=args.detector,
                                     beam_width=args.beam_width)
        step, lookback, lookahead, trim = args.segment_frames or 32, args.context_frames, 0, 0
    else:
        pipeline = load_eager_pipeline(args.checkpoint, args.sp_model_path, device=device,
                                       preprocess=args.preprocess, detector=args.detector,
                                       beam_width=args.beam_width)
        b = pipeline.backend
        step = args.segment_frames or b.segment_length
        lookback, lookahead, trim = args.context_frames, b.right_context_length, args.context_frames

    video, audio, fps = load_media_file(args.video)
    total = min(len(video), audio.size(0) // RATE_RATIO)
    media_sec = total / fps
    chunk_media = step / fps

    pre, front, dec, wall = [], [], [], []
    first_token_wall = None
    t_start = time.perf_counter()
    for i, (start, end, v_win, a_win) in enumerate(
        iter_av_windows(video, audio, step_frames=step, lookback_frames=lookback, lookahead_frames=lookahead)
    ):
        t0 = time.perf_counter()
        _, new, _ = pipeline.infer_chunk(v_win, a_win, trim_frames=trim)
        w = time.perf_counter() - t0
        if new.strip() and first_token_wall is None:
            first_token_wall = time.perf_counter() - t_start
        if i < args.warmup:
            continue  # exclude warm-up (kernel autotune, caches)
        tm = pipeline.last_timings
        pre.append(tm["preprocess"]); front.append(tm["frontend"]); dec.append(tm["decode"]); wall.append(w)

    if not wall:
        print("Not enough chunks after warmup; use a longer clip or smaller --warmup")
        return 1

    def stats(xs):
        return statistics.mean(xs), sorted(xs)[int(0.95 * (len(xs) - 1))]

    pre_m, _ = stats(pre); front_m, _ = stats(front); dec_m, _ = stats(dec)
    wall_m, wall_p95 = stats(wall)
    infer_m = front_m + dec_m

    print(f"\ndevice: {device}   clip: {os.path.basename(args.video)} "
          f"({total} frames @ {fps:.0f} fps = {media_sec:.1f}s)")
    print(f"chunks measured: {len(wall)} (excl. {args.warmup} warmup), {step} frames/chunk = {chunk_media*1000:.0f} ms media\n")
    print(f"  per-chunk mean (ms):  preprocess {pre_m*1e3:7.1f} | frontend {front_m*1e3:7.1f} | decode {dec_m*1e3:7.1f}")
    print(f"  per-chunk wall (ms):  mean {wall_m*1e3:7.1f}  p95 {wall_p95*1e3:7.1f}\n")
    print(f"  inference-only RTF (frontend+decode):     {infer_m/chunk_media:.3f}")
    print(f"  end-to-end   RTF (incl. preprocess):      {wall_m/chunk_media:.3f}")
    print(f"  preprocess share of end-to-end:           {pre_m/wall_m*100:.0f}%")
    print(f"  algorithmic latency (segment+rc)/fps:     {(step+lookahead)/fps*1000:.0f} ms")
    if first_token_wall is not None:
        print(f"  time-to-first-token (incl. model load excluded): {first_token_wall:.2f} s")
    print(f"\n  {'REAL-TIME ✓' if wall_m/chunk_media < 1.0 else 'NOT real-time ✗'} "
          f"(end-to-end RTF {'<' if wall_m/chunk_media < 1 else '>='} 1.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
