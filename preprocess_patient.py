#!/usr/bin/env python3
"""
Preprocess /scratch/th3482/LipVideoData/patient_25p short clips into mouth-crop datasets
for specified detectors, creating a train/val split with unseen validation sentences.

Rules:
- Input: 25 fps 1080p .mp4 files under /scratch/th3482/LipVideoData/patient_25p
- Split: Find the 10 least frequent sentences. For each, select one occurrence for VAL.
  Drop all other occurrences of those 10 sentences from both TRAIN and VAL so that
  validation sentences are unseen during training. Assign all remaining files to TRAIN.
- Transcript: filename.split('_')[0]
- No landmarks provided; detect on-the-fly; do not cache landmarks
- Save crops for each detector into separate roots for later training
- Full clips only (no segmenting); save video and audio crops + text
"""
import os
import glob
import argparse
import subprocess
from typing import Tuple, Dict, List
from collections import defaultdict
from tqdm import tqdm

import torch
import torchaudio
import torchvision

# Use repo loaders/processors
from preparation.data.data_module import AVSRDataLoader
from preparation.utils import save_vid_aud_txt
from preparation.transforms import TextTransform


def assign_splits(
    video_files: List[str],
    transcript_field: int = 0,
    val_sentence_count: int = 10,
) -> Dict[str, List[str]]:
    """
    Assign files to train/val ensuring validation sentences are unseen during training:
    - Find the least frequent sentences and choose up to `val_sentence_count` unique sentences
    - For each selected sentence, keep exactly ONE file in validation
    - Discard any other occurrences of these sentences from both splits
    - Assign all remaining files (from other sentences) to training
    """
    # 1. Extract transcripts and group files
    transcript_to_files = defaultdict(list)
    for path in video_files:
        transcript = parse_transcript_from_filename(path, transcript_field=transcript_field)
        transcript_to_files[transcript].append(path)
    
    # 2. Sort transcripts by frequency ASC (least frequent first)
    sorted_transcripts = sorted(
        transcript_to_files.items(),
        key=lambda x: (len(x[1]), x[0])  # fewest occurrences first, tie-break by text
    )
    
    # 3. Assign splits
    val_files = []
    train_files = []

    # 3a. Select up to val_sentence_count least frequent transcripts
    selected_for_val = []
    for transcript, files in sorted_transcripts:
        if len(selected_for_val) >= val_sentence_count:
            break
        selected_for_val.append((transcript, files))

    selected_transcripts = set(t for t, _ in selected_for_val)

    # 3b. For selected transcripts: keep exactly one instance in val, drop the rest
    for transcript, files in selected_for_val:
        if len(files) >= 1:
            val_files.append(files[0])
        # NOTE: intentionally DO NOT add remaining files to train to keep validation sentences unseen

    # 3c. For all other transcripts: add all files to training
    for transcript, files in sorted_transcripts:
        if transcript in selected_transcripts:
            continue
        train_files.extend(files)
    
    # Print statistics
    print("\nSplit Statistics:")
    print(f"Total files: {len(video_files)}")
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print("\nValidation sentences (least frequent) and counts:")
    for transcript, files in selected_for_val:
        print(f"- {transcript}: {len(files)}")
    
    return {
        "train": train_files,
        "val": val_files
    }


def select_shard_items(items: List[str], num_shards: int, shard_index: int) -> List[str]:
    """
    Split `items` into near-equal contiguous shards and return one shard.
    Example: 298 items, 6 shards -> [50, 50, 50, 50, 49, 49].
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards - 1}], got {shard_index}"
        )

    n_items = len(items)
    base = n_items // num_shards
    remainder = n_items % num_shards
    start = shard_index * base + min(shard_index, remainder)
    end = start + base + (1 if shard_index < remainder else 0)
    return items[start:end]


def parse_transcript_from_filename(path: str, transcript_field: int = 0) -> str:
    # transcript is an underscore-separated part selected by transcript_field
    # e.g., legacy file: "sentence_10-50_20250929_..." -> field 0 is sentence
    #       legal298 file: "100_sentence_10-50_repeat1_..." -> field 1 is sentence
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    if transcript_field < 0 or transcript_field >= len(parts):
        raise ValueError(
            f"transcript_field={transcript_field} out of range for filename: {name}"
        )
    # Convert transcript to uppercase for correct tokenization
    transcript = parts[transcript_field].strip().upper().replace(" ", "▁")
    return transcript


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_video_to_25fps(input_path: str, normalized_dir: str) -> str:
    """
    Re-encode one video to 25 fps.
    This also resolves orientation metadata differences across decoders.
    """
    ensure_dir(normalized_dir)
    output_path = os.path.join(normalized_dir, os.path.basename(input_path))
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        "fps=25",
        "-r",
        "25",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed for {input_path}: {err}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timeout for {input_path}") from e

    return output_path


def process_one(
    path: str,
    detector: str,
    out_root: str,
    text_transform: TextTransform,
    split: str,
    combine_av: bool,
    convert_gray: bool = False,
    transcript_field: int = 0,
    normalize_25fps: bool = False,
    normalized_dir: str = "",
) -> Tuple[bool, str, int, str, str, str]:
    transcript = parse_transcript_from_filename(path, transcript_field=transcript_field)
    input_video_path = path
    if normalize_25fps:
        input_video_path = normalize_video_to_25fps(path, normalized_dir)

    # Initialize loader per detector (video-only)
    vid_loader = AVSRDataLoader(modality="video", detector=detector, convert_gray=convert_gray)
    aud_loader = AVSRDataLoader(modality="audio")

    # Load video and crop via detector
    try:
        # Clear GPU cache before processing each video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        video_tensor = vid_loader.load_data(input_video_path, landmarks=None)
        audio_tensor = aud_loader.load_data(input_video_path)
        
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
    dataset_name = f"patient_{detector}"
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
    parser = argparse.ArgumentParser(description="Preprocess patient_25p into cropped datasets for two detectors")
    parser.add_argument("--input_dir", type=str, default="/scratch/th3482/LipVideoData/patient_25p")
    parser.add_argument("--output_root", type=str, default="/scratch/th3482/LipVideoData/patient_25p_crops")
    parser.add_argument("--detectors", type=str, default="retinaface,mediapipe")
    parser.add_argument("--convert_gray", action="store_true", help="Convert crops to grayscale")
    parser.add_argument("--combine_av", action="store_true", help="Mux audio into mp4 and remove wav")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of videos to process before clearing cache")
    parser.add_argument(
        "--transcript_field",
        type=int,
        default=0,
        help="Index of underscore-separated filename field that contains transcript text",
    )
    parser.add_argument(
        "--val_sentence_count",
        type=int,
        default=10,
        help="Number of least-frequent sentences to hold out for validation (unseen in train)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards used to process assigned files in parallel",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based index of this shard in [0, num_shards-1]",
    )
    parser.add_argument(
        "--label_suffix",
        type=str,
        default="",
        help="Optional suffix inserted before .csv for label files (e.g., _part0)",
    )
    parser.add_argument(
        "--normalize_25fps",
        action="store_true",
        help="Normalize each input video to 25 fps before landmark detection/cropping",
    )
    parser.add_argument(
        "--normalized_dir",
        type=str,
        default="",
        help="Directory to cache normalized 25 fps videos; defaults to <output_root>/_normalized_25p",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    out_root = args.output_root
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    normalized_dir = args.normalized_dir or os.path.join(out_root, "_normalized_25p")

    ensure_dir(out_root)
    text_transform = TextTransform()
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    print(f"Found {len(video_files)} mp4 files under {input_dir}")

    # 基于重复次数分配训练/验证集
    splits = assign_splits(
        video_files,
        transcript_field=args.transcript_field,
        val_sentence_count=args.val_sentence_count,
    )

    # 创建一个文件到分割集的映射，方便后续查找
    file_to_split = {f: s for s, files in splits.items() for f in files}

    for detector in detectors:
        ok, fail = 0, 0
        # Prepare label CSVs per split
        dataset_name = f"patient_{detector}"
        labels_dir = os.path.join(out_root, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        label_files = {
            "train": os.path.join(labels_dir, f"{dataset_name}_train_transcript_lengths_seg24s{args.label_suffix}.csv"),
            "val": os.path.join(labels_dir, f"{dataset_name}_val_transcript_lengths_seg24s{args.label_suffix}.csv"),
        }
        # Open files in append mode to preserve progress if killed
        handles = {split: open(path, "a") for split, path in label_files.items()}
        print(f"\n=== Processing with detector: {detector} ===")
        # Only process files assigned to a split (skip dropped duplicates)
        assigned_files = splits["train"] + splits["val"]
        if args.num_shards > 1:
            assigned_files = select_shard_items(
                assigned_files, num_shards=args.num_shards, shard_index=args.shard_index
            )
            print(
                f"Shard {args.shard_index}/{args.num_shards}: processing {len(assigned_files)} assigned files"
            )
        for i, path in enumerate(tqdm(assigned_files, desc=f"{detector}")):
            split = file_to_split[path]
            success, split, nframes, token_ids, ds_name, rel_base = process_one(
                path,
                detector,
                out_root,
                text_transform,
                split,
                args.combine_av,
                convert_gray=args.convert_gray,
                transcript_field=args.transcript_field,
                normalize_25fps=args.normalize_25fps,
                normalized_dir=normalized_dir,
            )
            if success:
                ok += 1
                if token_ids and rel_base:
                    # Write CSV line: dataset_name,relative_path,num_frames,token_ids
                    # - dataset_name: e.g., "patient_retinaface"
                    # - relative_path: path relative to dataset root, e.g., "patient_retinaface_video_seg24s/train/xxx.mp4"
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
