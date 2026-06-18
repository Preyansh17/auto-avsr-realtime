#!/usr/bin/env python3
"""Evaluate a streaming AVSR checkpoint on patient data.

Modes:
  utterance  full-sequence beam search per clip (offline upper bound)
  streaming  chunked StreamingInferencePipeline, same path as
             demo_realtime.py, reporting WER and per-chunk RTF

Data: --test-file label CSV (1023-piece token ids) under --root-dir, or
--patient-dir to scan a directory of videos (references from filenames).
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.checkpoint import preflight_environment  # noqa: E402
from online_avsr.data_module import PatientAVCollate  # noqa: E402
from online_avsr.patient_dataset import (  # noqa: E402
    PatientAVDataset,
    discover_patient_records,
)
from online_avsr.streaming import (  # noqa: E402
    RATE_RATIO,
    iter_av_windows,
    load_eager_pipeline,
    load_jit_pipeline,
    load_media_file,
)
from online_avsr.text import compute_wer, load_sentencepiece_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--checkpoint")
    model.add_argument("--jit-model")
    parser.add_argument("--mode", choices=["utterance", "streaming"], default="streaming")
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--test-file", default=None, help="Label CSV (requires --root-dir)")
    parser.add_argument("--patient-dir", default=None, help="Directory of videos (references from filenames)")
    parser.add_argument("--sp-model-path", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"))
    parser.add_argument("--preprocess", choices=["face", "mouth", "roi", "none"], default=None)
    parser.add_argument("--detector", choices=["mediapipe", "retinaface"], default="mediapipe")
    parser.add_argument("--segment-frames", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None, help="Skip clips longer than this")
    parser.add_argument("--leading-silence-frames", type=int, default=0,
                        help="Prepend N silent frames to each clip before streaming to warm up the Emformer")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "eval"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_records(args, sp_model):
    if args.test_file:
        if not args.root_dir:
            raise SystemExit("--test-file requires --root-dir")
        records = discover_patient_records(args.root_dir, label_file=args.test_file, sp_model=sp_model)
    elif args.patient_dir:
        records = discover_patient_records(args.patient_dir)
    else:
        raise SystemExit("Provide --test-file with --root-dir, or --patient-dir")
    if args.max_frames:
        records = [r for r in records if not r.frames or r.frames <= args.max_frames]
    if args.limit:
        records = records[: args.limit]
    return records


def eval_streaming(args, pipeline, records, step, lookback, lookahead, trim, jsonl_path):
    results = []
    with open(jsonl_path, "a", encoding="utf-8") as jsonl:
        for record in records:
            try:
                video, audio, fps = load_media_file(record.path)
                if args.leading_silence_frames > 0:
                    n = args.leading_silence_frames
                    video = np.concatenate([np.zeros((n, *video.shape[1:]), dtype=video.dtype), video])
                    audio = torch.cat([torch.zeros(n * RATE_RATIO, audio.size(1), dtype=audio.dtype), audio])
                pipeline.reset()
                walls = []
                for start, end, v_win, a_win in iter_av_windows(
                    video, audio, step_frames=step, lookback_frames=lookback, lookahead_frames=lookahead
                ):
                    t0 = time.perf_counter()
                    transcript, _, _ = pipeline.infer_chunk(v_win, a_win, trim_frames=trim)
                    walls.append(time.perf_counter() - t0)
                media_sec = min(len(video), audio.size(0) // 640) / fps
                transcript = pipeline.transcript.strip()
                result = {
                    "event": "final",
                    "video_path": record.path,
                    "reference": record.text,
                    "transcript": transcript,
                    "wer": compute_wer(record.text, transcript) if record.text else None,
                    "chunks": len(walls),
                    "media_sec": media_sec,
                    "decode_wall_sec": sum(walls),
                    "rtf": sum(walls) / media_sec if media_sec else None,
                }
            except Exception as exc:  # keep evaluating the rest of the set
                result = {"event": "error", "video_path": record.path, "error": str(exc)}
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
            tag = f"WER={result.get('wer'):.3f}" if result.get("wer") is not None else result["event"]
            print(f"[{len(results)}/{len(records)}] {os.path.basename(record.path)}: {tag}")
    return results


def eval_utterance(args, module, sp_model, records, frame_size, jsonl_path):
    from online_avsr.module import AVBatch  # noqa: F401  (collate returns AVBatch)

    dataset = PatientAVDataset(
        args.root_dir or args.patient_dir,
        label_file=args.test_file if args.test_file else None,
        sp_model=sp_model,
        limit=args.limit,
        max_frames=args.max_frames,
    )
    collate = PatientAVCollate(sp_model, "test", frame_size=frame_size)
    results = []
    with open(jsonl_path, "a", encoding="utf-8") as jsonl:
        for i in range(len(dataset)):
            sample = dataset[i]
            try:
                batch = collate([sample])
                t0 = time.perf_counter()
                with torch.inference_mode():
                    transcript = module(batch)
                wall = time.perf_counter() - t0
                media_sec = batch.video_lengths[0].item() / 25.0
                result = {
                    "event": "final",
                    "video_path": sample["path"],
                    "reference": sample["text"],
                    "transcript": transcript,
                    "wer": compute_wer(sample["text"], transcript) if sample["text"] else None,
                    "media_sec": media_sec,
                    "decode_wall_sec": wall,
                    "rtf": wall / media_sec if media_sec else None,
                }
            except Exception as exc:
                result = {"event": "error", "video_path": sample["path"], "error": str(exc)}
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)
            tag = f"WER={result.get('wer'):.3f}" if result.get("wer") is not None else result["event"]
            print(f"[{len(results)}/{len(dataset)}] {os.path.basename(sample['path'])}: {tag}")
    return results


def main():
    args = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0"
    sp_model = load_sentencepiece_model(args.sp_model_path)
    records = collect_records(args, sp_model)
    if not records:
        raise SystemExit("No evaluation records found")

    if args.dry_run:
        print(json.dumps({
            "preflight": preflight_environment(),
            "mode": args.mode,
            "records": len(records),
            "first": records[0].path,
            "model": args.checkpoint or args.jit_model,
        }, indent=2))
        return

    run_dir = os.path.join(args.output_dir, f"eval_{args.mode}_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    jsonl_path = os.path.join(run_dir, "results.jsonl")

    if args.jit_model:
        pipeline = load_jit_pipeline(
            args.jit_model, args.sp_model_path, device=device,
            preprocess=args.preprocess or "face", detector=args.detector, beam_width=args.beam_width,
        )
        step, lookback, lookahead, trim = args.segment_frames or 32, args.context_frames, 0, 0
        module = None
        frame_size = 44
    else:
        pipeline = load_eager_pipeline(
            args.checkpoint, args.sp_model_path, device=device,
            preprocess=args.preprocess, detector=args.detector, beam_width=args.beam_width,
        )
        backend = pipeline.backend
        module = backend.module
        frame_size = 44 if module.architecture == "device" else 88
        step = args.segment_frames or backend.segment_length
        lookback, lookahead, trim = args.context_frames, backend.right_context_length, args.context_frames

    if args.mode == "streaming":
        results = eval_streaming(args, pipeline, records, step, lookback, lookahead, trim, jsonl_path)
    else:
        if module is None:
            raise SystemExit("--mode utterance requires --checkpoint (JIT model is streaming-only)")
        results = eval_utterance(args, module, sp_model, records, frame_size, jsonl_path)

    ok = [r for r in results if r["event"] == "final"]
    wers = [r["wer"] for r in ok if r.get("wer") is not None]
    rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]
    summary = {
        "mode": args.mode,
        "model": args.checkpoint or args.jit_model,
        "videos_total": len(records),
        "videos_ok": len(ok),
        "avg_wer": statistics.mean(wers) if wers else None,
        "avg_rtf": statistics.mean(rtfs) if rtfs else None,
        "results_jsonl": jsonl_path,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
