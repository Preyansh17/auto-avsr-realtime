#!/usr/bin/env python3
"""Pre-bake face crops from original patient videos, matching an existing split.

The pretrained device_avsr video frontend is a Linear layer trained on 44x44
*face* crops; the patient data is mouth ROIs. This script reproduces face crops
on the SAME split: it iterates a label CSV (which defines the split and the
per-clip rel_paths), finds each clip's ORIGINAL video under --orig-root, runs
the device_avsr face pipeline (mediapipe detect -> align to mean face -> crop ->
96x96), and writes a face-crop mp4 (+ sibling 16 kHz wav) under --out-root with
the SAME dataset_name/rel_path. Training/eval then point PATIENT_DATA_ROOT at
--out-root and use --preprocess roi (resize 96->44, grayscale, normalize) --
equivalent to --preprocess face but precomputed once.

Filename matching (--match):
  basename  glob --orig-root for a file whose basename == the clip's basename
            (default; use when originals are per-utterance with the same name)
  relpath   --orig-root/<rel_path> (originals mirror the label dir structure)

Usage:
  python scripts/face_crop_patient.py \
    --label-csv /scratch/.../labels/patient_retinaface_train_transcript_lengths_seg24s.csv \
    --orig-root /scratch/th3482/LipVideoData/patient_legal298_crops_unseen/_normalized_25p \
    --out-root  /scratch/pa2753/LipVideoData/patient_legal298_facecrop
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torchaudio
import torchvision

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-csv", required=True, nargs="+", help="Label CSV(s) defining the split + rel_paths")
    parser.add_argument("--orig-root", required=True, help="Root of original (full-frame) patient videos")
    parser.add_argument("--out-root", required=True, help="Where to write face-crop mp4s (mirrors dataset_name/rel_path)")
    parser.add_argument("--match", choices=["basename", "relpath"], default="basename")
    parser.add_argument("--detector", choices=["mediapipe", "retinaface"], default="mediapipe")
    parser.add_argument("--fps", type=int, default=25, help="Output video fps")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve originals + report coverage, write nothing")
    return parser.parse_args()


def build_face_processor(detector):
    if detector == "mediapipe":
        from preparation.detectors.mediapipe_face.detector import LandmarksDetector
        from preparation.detectors.mediapipe_face.video_process import VideoProcess

        return LandmarksDetector(), VideoProcess()
    from preparation.detectors.retinaface.detector import LandmarksDetector
    from preparation.detectors.retinaface.video_process import VideoProcess

    return LandmarksDetector(device="cpu"), VideoProcess(convert_gray=False)


def iter_rows(label_csv):
    with open(label_csv, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            yield parts[0], parts[1]  # dataset_name, rel_path


def find_original(rel_path, orig_root, match, _cache={}):
    if match == "relpath":
        cand = os.path.join(orig_root, rel_path)
        return cand if os.path.isfile(cand) else None
    # basename: build an index once, then look up
    if "index" not in _cache:
        idx = {}
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
            for p in glob.glob(os.path.join(orig_root, "**", ext), recursive=True):
                idx.setdefault(os.path.basename(p), p)  # first match wins
        _cache["index"] = idx
    return _cache["index"].get(os.path.basename(rel_path))


def load_video_audio(path):
    v, a, info = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
    video = v.numpy().astype(np.uint8)
    if a.numel() > 0:
        audio, sr = a.float(), int(info.get("audio_fps") or 16000)
    else:
        wav = os.path.splitext(path)[0] + ".wav"
        if os.path.isfile(wav):
            audio, sr = torchaudio.load(wav)
        else:
            audio, sr = None, None
    if audio is not None and sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    return video, audio  # audio: (channels, N) or None


def main():
    args = parse_args()
    detector, video_process = (None, None) if args.dry_run else build_face_processor(args.detector)

    found = missing = cropped = failed = 0
    seen = set()
    for label_csv in args.label_csv:
        for dataset_name, rel_path in iter_rows(label_csv):
            if (dataset_name, rel_path) in seen:
                continue
            seen.add((dataset_name, rel_path))
            if args.limit and (found + missing) >= args.limit:
                break

            src = find_original(rel_path, args.orig_root, args.match)
            if not src:
                missing += 1
                if missing <= 10:
                    print(f"  MISSING original for {rel_path}", file=sys.stderr)
                continue
            found += 1

            out_mp4 = os.path.join(args.out_root, dataset_name, rel_path)
            if not args.overwrite and os.path.isfile(out_mp4):
                continue
            if args.dry_run:
                continue

            try:
                video, audio = load_video_audio(src)
                landmarks = detector(video)
                face = video_process(video, landmarks)
                if face is None:
                    raise RuntimeError("no face landmarks")
                face = torch.from_numpy(np.ascontiguousarray(face)).to(torch.uint8)  # T,H,W,C (96x96)
                os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
                torchvision.io.write_video(out_mp4, face, fps=args.fps)
                if audio is not None:
                    wav = os.path.splitext(out_mp4)[0] + ".wav"
                    torchaudio.save(wav, audio, 16000)
                cropped += 1
                if cropped % 25 == 0:
                    print(f"  cropped {cropped} ...")
            except Exception as exc:
                failed += 1
                print(f"  FAILED {rel_path}: {exc}", file=sys.stderr)

    print(f"\noriginals found={found} missing={missing} | cropped={cropped} failed={failed}")
    if missing:
        print(f"WARNING: {missing} clips had no original under {args.orig_root} "
              f"(try --match relpath, or check the directory)", file=sys.stderr)
    return 0 if found and not (args.dry_run is False and cropped == 0 and found) else 0


if __name__ == "__main__":
    sys.exit(main())
