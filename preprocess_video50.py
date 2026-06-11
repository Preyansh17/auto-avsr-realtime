#!/usr/bin/env python3
"""
Preprocess /scratch/th3482/LipVideoData/video_50_25p short clips into mouth-crop datasets
for both retinaface and mediapipe detectors, splitting into train/val/test by repeatN in filename.

Rules:
- Input: 25 fps 1080p .mp4 files under /scratch/th3482/LipVideoData/video_50_25p
- Split: repeat1,2,4,5 -> train; repeat6 -> val; repeat3 -> test
- Transcript: filename.split('_')[0]
- No landmarks provided; detect on-the-fly; do not cache landmarks
- Save crops for both detectors into separate roots for later training
- Full clips only (no segmenting); save video-only crops + text (audio optional)
"""
import os
import re
import glob
import argparse
from typing import Tuple
from tqdm import tqdm

import torch
import torchaudio
import torchvision

# Use repo loaders/processors
from preparation.data.data_module import AVSRDataLoader
from preparation.utils import save_vid_aud_txt
from preparation.transforms import TextTransform


def parse_split_from_filename(path: str) -> str:
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    # e.g., "..._repeat5_2025...."
    m = re.search(r"repeat(\d+)", stem)
    if not m:
        return "train"  # default to train if missing
    rep = int(m.group(1))
    if rep in {1, 2, 4, 5}:
        return "train"
    if rep == 6:
        return "val"
    if rep == 3:
        return "test"
    return "train"


def parse_transcript_from_filename(path: str) -> str:
    # transcript is the first underscore-separated part
    # e.g., "But I do not know the schedule for the trial_10-50_repeat1_..."
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    # Convert transcript to uppercase for correct tokenization
    transcript = parts[0].strip().upper().replace(" ", "▁")
    return transcript


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_one(path: str, detector: str, out_root: str, text_transform: TextTransform, combine_av: bool, convert_gray: bool = False) -> Tuple[bool, str, int, str, str, str]:
    split = parse_split_from_filename(path)
    transcript = parse_transcript_from_filename(path)
    print(transcript)

    # Initialize loader per detector (video-only)
    vid_loader = AVSRDataLoader(modality="video", detector=detector, convert_gray=convert_gray)
    aud_loader = AVSRDataLoader(modality="audio")

    # Load video and crop via detector
    try:
        # Clear GPU cache before processing each video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        video_tensor = vid_loader.load_data(path, landmarks=None)
        audio_tensor = aud_loader.load_data(path)
        
        # Move tensors to CPU to save GPU memory
        if video_tensor.is_cuda:
            video_tensor = video_tensor.cpu()
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
    except Exception as e:
        print(f"[SKIP] {path} due to detector/crop error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, split, 0, "", "", ""

    # Dataset-style output paths mirroring original preprocessor
    dataset_name = f"video50_{detector}"
    dst_vid_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_video_seg24s", split)
    dst_txt_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_text_seg24s", split)

    rel_name = os.path.basename(path)
    stem = os.path.splitext(rel_name)[0]
    dst_vid = os.path.join(dst_vid_dir, stem + ".mp4")
    dst_aud = os.path.join(dst_vid_dir, stem + ".wav")
    dst_txt = os.path.join(dst_txt_dir, stem + ".txt")

    num_frames = int(video_tensor.shape[0])
    token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(transcript)]))

    try:
        save_vid_aud_txt(
            dst_vid,
            dst_aud,
            dst_txt,
            video_tensor,
            audio_tensor,
            transcript,
            video_fps=25,
            audio_sample_rate=16000,
        )
        # Optional: mux audio into mp4 like original combine-av
        if combine_av:
            import ffmpeg, shutil
            in1 = ffmpeg.input(dst_vid)
            in2 = ffmpeg.input(dst_aud)
            out = ffmpeg.output(
                in1["v"],
                in2["a"],
                dst_vid[:-4] + ".av.mp4",
                vcodec="copy",
                acodec="aac",
                strict="experimental",
                loglevel="panic",
            )
            out.run()
            try:
                os.remove(dst_aud)
            except Exception:
                pass
            shutil.move(dst_vid[:-4] + ".av.mp4", dst_vid)
        print(f"[OK] {detector} {split} -> {dst_vid}")
        # Return info for labels CSV
        rel_base = os.path.relpath(dst_vid, start=os.path.join(out_root, dataset_name))
        return True, split, num_frames, token_id_str, dataset_name, rel_base
    except Exception as e:
        print(f"[FAIL] Save failed for {path}: {e}")
        return False, split, 0, "", "", ""


def main():
    parser = argparse.ArgumentParser(description="Preprocess video_50_25p into cropped datasets for two detectors")
    parser.add_argument("--input_dir", type=str, default="/scratch/th3482/LipVideoData/video_50_25p")
    parser.add_argument("--output_root", type=str, default="/scratch/th3482/LipVideoData/video_50_25p_crops")
    parser.add_argument("--detectors", type=str, default="retinaface,mediapipe")
    parser.add_argument("--convert_gray", action="store_true", help="Convert crops to grayscale")
    parser.add_argument("--combine_av", action="store_true", help="Mux audio into mp4 and remove wav")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of videos to process before clearing cache")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_root = args.output_root
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    ensure_dir(out_root)
    text_transform = TextTransform()
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    print(f"Found {len(video_files)} mp4 files under {input_dir}")

    for detector in detectors:
        ok, fail = 0, 0
        # Prepare label CSVs per split
        dataset_name = f"video50_{detector}"
        labels_dir = os.path.join(out_root, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        label_files = {
            "train": os.path.join(labels_dir, f"{dataset_name}_train_transcript_lengths_seg24s.csv"),
            "val": os.path.join(labels_dir, f"{dataset_name}_val_transcript_lengths_seg24s.csv"),
            "test": os.path.join(labels_dir, f"{dataset_name}_test_transcript_lengths_seg24s.csv"),
        }
        # Open files in append mode to preserve progress if killed
        handles = {split: open(path, "a") for split, path in label_files.items()}
        print(f"\n=== Processing with detector: {detector} ===")
        for i, path in enumerate(tqdm(video_files, desc=f"{detector}")):
            success, split, nframes, token_ids, ds_name, rel_base = process_one(
                path, detector, out_root, text_transform, args.combine_av, convert_gray=args.convert_gray
            )
            if success:
                ok += 1
                if token_ids and rel_base:
                    # Write CSV line: dataset_name,relative_path,num_frames,token_ids
                    # - dataset_name: e.g., "video50_retinaface"
                    # - relative_path: path relative to dataset root, e.g., "video50_retinaface_video_seg24s/train/xxx.mp4"
                    # - num_frames: number of frames in the video
                    # - token_ids: space-separated token IDs from text_transform.tokenize()
                    handles[split].write(f"{ds_name},{rel_base},{nframes},{token_ids}\n")
                    handles[split].flush()
                    os.fsync(handles[split].fileno())  # Force write to disk
            else:
                fail += 1
            
            # Clear memory every batch_size videos
            if (i + 1) % args.batch_size == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                # Also flush all files periodically just in case
                for h in handles.values():
                    h.flush()
                    os.fsync(h.fileno())
        print(f"Detector {detector}: OK={ok} FAIL={fail}")
        for h in handles.values():
            h.flush()
            os.fsync(h.fileno())
            h.close()


if __name__ == "__main__":
    main()


