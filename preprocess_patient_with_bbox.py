#!/usr/bin/env python3
"""
Preprocess /scratch/th3482/LipVideoData/patient_25p short clips into:
1. Mouth-crop datasets with audio
2. Original videos with blue bounding boxes around mouth with audio

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
import re
import glob
import argparse
from typing import Tuple, Dict, List
from collections import defaultdict
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torchaudio
import torchvision
import ffmpeg

# Use repo loaders/processors
from preparation.data.data_module import AVSRDataLoader
from preparation.utils import save_vid_aud_txt
from preparation.transforms import TextTransform


def assign_splits(video_files: List[str], val_sentence_count: int = 10) -> Dict[str, List[str]]:
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
        transcript = parse_transcript_from_filename(path)
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


def parse_transcript_from_filename(path: str) -> str:
    # transcript is the first underscore-separated part
    # e.g., "But I do not know the schedule for the trial_10-50_20250929_..."
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    # Convert transcript to uppercase for correct tokenization
    transcript = parts[0].strip().upper().replace(" ", "▁")
    return transcript


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_bbox_on_video(video_frames, all_landmarks, video_processor):
    """
    Draw blue bounding boxes on original video frames that match the actual crop regions.
    This simulates the affine transformation and cropping process to show the exact region being cropped.
    """
    annotated_frames = []
    
    # Preprocess landmarks (same as video_process does)
    preprocessed_landmarks = video_processor.interpolate_landmarks(all_landmarks.copy())
    if not preprocessed_landmarks:
        # If preprocessing failed, return original frames
        return video_frames
    
    for frame_idx, frame in enumerate(video_frames):
        frame_copy = frame.copy()
        
        # Calculate smoothed landmarks (same as in video_process.crop_patch)
        window_margin = min(
            video_processor.window_margin // 2, 
            frame_idx, 
            len(preprocessed_landmarks) - 1 - frame_idx
        )
        smoothed_landmarks = np.mean(
            [
                preprocessed_landmarks[x]
                for x in range(
                    frame_idx - window_margin, frame_idx + window_margin + 1
                )
            ],
            axis=0,
        )
        smoothed_landmarks += preprocessed_landmarks[frame_idx].mean(
            axis=0
        ) - smoothed_landmarks.mean(axis=0)
        
        # Perform affine transformation to get transformed coordinates
        try:
            # Use the video_processor's own affine_transform method to get the exact transform
            # This ensures we use the same parameters as the actual cropping process
            transformed_frame, all_transformed_landmarks = video_processor.affine_transform(
                frame, 
                smoothed_landmarks, 
                video_processor.reference, 
                grayscale=video_processor.convert_gray
            )
            
            # Get the transform matrix that was used
            # We need to re-compute it to get the matrix itself
            if video_processor.start_idx == 48:  # retinaface
                stable_points = (28, 33, 36, 39, 42, 45, 48, 54)
                stable_reference = video_processor.get_stable_reference(
                    video_processor.reference, stable_points, (256, 256), (256, 256)
                )
            else:  # mediapipe
                stable_points = (0, 1, 2, 3)
                stable_reference = video_processor.get_stable_reference(
                    video_processor.reference, (256, 256), (256, 256)
                )
            
            transform = video_processor.estimate_affine_transform(
                smoothed_landmarks, stable_points, stable_reference
            )
            
            # Now get the mouth landmarks from transformed space
            transformed_mouth_landmarks = all_transformed_landmarks[video_processor.start_idx : video_processor.stop_idx]
            
            # Calculate the crop box center in transformed space (same as cut_patch does)
            center_x, center_y = np.mean(transformed_mouth_landmarks, axis=0)
            crop_half_width = video_processor.crop_width // 2
            crop_half_height = video_processor.crop_height // 2
            
            # Calculate crop boundaries exactly as cut_patch does
            x_min = center_x - crop_half_width
            x_max = center_x + crop_half_width
            y_min = center_y - crop_half_height
            y_max = center_y + crop_half_height
            
            # Create the four corners of the crop box in transformed space
            crop_corners_transformed = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ])
            
            # Inverse transform to get back to original image space
            inv_transform = cv2.invertAffineTransform(transform)
            crop_corners_for_transform = crop_corners_transformed.reshape(-1, 1, 2).astype(np.float32)
            crop_corners_original_cv = cv2.transform(crop_corners_for_transform, inv_transform)
            crop_corners_original = crop_corners_original_cv.reshape(-1, 2)
            
            # Draw the polygon representing the crop region
            # Use cyan color (RGB format) for a clean, tech-style look
            pts = crop_corners_original.astype(np.int32)
            cv2.polylines(frame_copy, [pts], isClosed=True, color=(0, 255, 255), thickness=3)
                
        except Exception as e:
            # If transform fails, fall back to simple bounding box around mouth
            mouth_landmarks = smoothed_landmarks[video_processor.start_idx : video_processor.stop_idx]
            center_x, center_y = np.mean(mouth_landmarks, axis=0)
            # Use approximate box size
            box_size = 120  # Approximate size in original space
            x_min = int(max(0, center_x - box_size))
            x_max = int(min(frame.shape[1], center_x + box_size))
            y_min = int(max(0, center_y - box_size))
            y_max = int(min(frame.shape[0], center_y + box_size))
            cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
        
        annotated_frames.append(frame_copy)
    
    return np.array(annotated_frames)


def save_video_with_audio(video_path, audio_path, video_tensor, audio_tensor, video_fps=25, audio_sample_rate=16000):
    """
    Save video and audio, then mux them together
    """
    ensure_dir(os.path.dirname(video_path))
    ensure_dir(os.path.dirname(audio_path))
    
    # Save video (without audio)
    video_tensor_uint8 = video_tensor.to(torch.uint8)
    torchvision.io.write_video(
        video_path.replace('.mp4', '_temp.mp4'),
        video_tensor_uint8,
        fps=video_fps,
        video_codec='libx264'
    )
    
    # Save audio
    torchaudio.save(audio_path, audio_tensor, audio_sample_rate)
    
    # Mux video and audio using ffmpeg
    try:
        in1 = ffmpeg.input(video_path.replace('.mp4', '_temp.mp4'))
        in2 = ffmpeg.input(audio_path)
        out = ffmpeg.output(
            in1["v"],
            in2["a"],
            video_path,
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            loglevel="panic",
        )
        out.run(overwrite_output=True)
        # Clean up temporary files
        os.remove(video_path.replace('.mp4', '_temp.mp4'))
    except Exception as e:
        print(f"Warning: Failed to mux audio: {e}")
        # If muxing fails, at least keep the video
        import shutil
        shutil.move(video_path.replace('.mp4', '_temp.mp4'), video_path)


def process_one(path: str, detector: str, out_root: str, text_transform: TextTransform, split: str, combine_av: bool, convert_gray: bool = False) -> Tuple[bool, str, int, str, str, str]:
    transcript = parse_transcript_from_filename(path)

    # Initialize loader per detector (video-only)
    vid_loader = AVSRDataLoader(modality="video", detector=detector, convert_gray=convert_gray)
    aud_loader = AVSRDataLoader(modality="audio")

    # Load original video for bbox version
    original_video = torchvision.io.read_video(path, pts_unit="sec")[0].numpy()
    
    # Load video and crop via detector
    try:
        # Clear GPU cache before processing each video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get landmarks for both cropped and bbox versions
        landmarks = vid_loader.landmarks_detector(original_video)
        
        # Create cropped video
        video_tensor = torch.tensor(vid_loader.video_process(original_video, landmarks))
        audio_tensor = aud_loader.load_data(path)
        
        # Create bbox version - pass video_processor to draw accurate crop regions
        bbox_video = draw_bbox_on_video(original_video, landmarks, vid_loader.video_process)
        bbox_video_tensor = torch.from_numpy(bbox_video)
        
        # Move tensors to CPU to save GPU memory
        if video_tensor.is_cuda:
            video_tensor = video_tensor.cpu()
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
        if bbox_video_tensor.is_cuda:
            bbox_video_tensor = bbox_video_tensor.cpu()
    except Exception as e:
        print(f"[SKIP] {path} due to detector/crop error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, split, 0, "", "", ""

    # Dataset-style output paths
    dataset_name = f"patient_{detector}"
    
    # Mouth crop paths
    dst_vid_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_video_seg24s", split)
    dst_txt_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_text_seg24s", split)
    
    # Original with bbox paths
    dst_bbox_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_video_bbox_seg24s", split)

    rel_name = os.path.basename(path)
    stem = os.path.splitext(rel_name)[0]
    
    # Mouth crop files
    dst_vid = os.path.join(dst_vid_dir, stem + ".mp4")
    dst_aud = os.path.join(dst_vid_dir, stem + ".wav")
    dst_txt = os.path.join(dst_txt_dir, stem + ".txt")
    
    # Bbox version files
    dst_bbox_vid = os.path.join(dst_bbox_dir, stem + ".mp4")
    dst_bbox_aud = os.path.join(dst_bbox_dir, stem + ".wav")

    num_frames = int(video_tensor.shape[0])
    token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(transcript)]))

    try:
        # Save mouth crop version
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
        
        # Optional: mux audio into mp4 for mouth crop
        if combine_av:
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
            out.run(overwrite_output=True)
            try:
                os.remove(dst_aud)
            except Exception:
                pass
            import shutil
            shutil.move(dst_vid[:-4] + ".av.mp4", dst_vid)
        
        # Save bbox version with audio
        save_video_with_audio(
            dst_bbox_vid,
            dst_bbox_aud,
            bbox_video_tensor,
            audio_tensor,
            video_fps=25,
            audio_sample_rate=16000
        )
        
        # Remove separate wav file for bbox if combine_av is True
        if combine_av:
            try:
                os.remove(dst_bbox_aud)
            except Exception:
                pass
        
        print(f"[OK] {detector} {split} -> {dst_vid} + {dst_bbox_vid}")
        
        # Return info for labels CSV
        rel_base = os.path.relpath(dst_vid, start=os.path.join(out_root, dataset_name))
        return True, split, num_frames, token_id_str, dataset_name, rel_base
    except Exception as e:
        print(f"[FAIL] Save failed for {path}: {e}")
        import traceback
        traceback.print_exc()
        return False, split, 0, "", "", ""


def main():
    parser = argparse.ArgumentParser(description="Preprocess patient_25p into cropped datasets + bbox videos")
    parser.add_argument("--input_dir", type=str, default="/scratch/th3482/LipVideoData/patient_25p")
    parser.add_argument("--output_root", type=str, default="/scratch/th3482/LipVideoData/patient_25p_crops_bbox")
    parser.add_argument("--detectors", type=str, default="retinaface")
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

    # 基于重复次数分配训练/验证集
    splits = assign_splits(video_files, val_sentence_count=10)

    # Only process test (validation) set
    test_files = splits["val"]
    print(f"\nProcessing only TEST set: {len(test_files)} files")

    for detector in detectors:
        ok, fail = 0, 0
        # Prepare label CSV for test split only
        dataset_name = f"patient_{detector}"
        labels_dir = os.path.join(out_root, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        label_file = os.path.join(labels_dir, f"{dataset_name}_test_transcript_lengths_seg24s.csv")
        # Open file in append mode to preserve progress if killed
        handle = open(label_file, "a")
        print(f"\n=== Processing with detector: {detector} ===")
        # Only process test files
        for i, path in enumerate(tqdm(test_files, desc=f"{detector}")):
            split = "test"
            success, split, nframes, token_ids, ds_name, rel_base = process_one(
                path, detector, out_root, text_transform, split, args.combine_av, convert_gray=args.convert_gray
            )
            if success:
                ok += 1
                if token_ids and rel_base:
                    # Write CSV line: dataset_name,relative_path,num_frames,token_ids
                    handle.write(f"{ds_name},{rel_base},{nframes},{token_ids}\n")
                    handle.flush()
                    os.fsync(handle.fileno())  # Force write to disk
            else:
                fail += 1
            
            # Clear memory every batch_size videos
            if (i + 1) % args.batch_size == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                # Also flush periodically
                handle.flush()
                os.fsync(handle.fileno())
        print(f"Detector {detector}: OK={ok} FAIL={fail}")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()


if __name__ == "__main__":
    main()

