#!/usr/bin/env python3
"""
Preprocess /scratch/th3482/LipVideoData/video_300_25p short clips into:
1. Mouth-crop datasets with audio
2. Original videos with blue bounding boxes around mouth with audio

Rules:
- Input: 25 fps 1080p .mp4 files under /scratch/th3482/LipVideoData/video_300_25p
- Split: Random split with 200 train, 50 val, 50 test
- Transcript: filename.split('_')[0]
- No landmarks provided; detect on-the-fly; do not cache landmarks
- Save crops for both detectors into separate roots for later training
- Full clips only (no segmenting); save video-only crops + text (audio optional)
"""
import os
import re
import glob
import random
import argparse
from typing import Tuple, Dict, List
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


def create_random_split(video_files: List[str], split_sizes: Dict[str, int], seed: int = 42) -> Dict[str, List[str]]:
    """
    Randomly split video files into train/val/test sets with specified sizes.
    """
    random.seed(seed)
    files = video_files.copy()
    random.shuffle(files)
    
    splits = {}
    start_idx = 0
    for split_name, size in split_sizes.items():
        end_idx = start_idx + size
        splits[split_name] = files[start_idx:end_idx]
        start_idx = end_idx
    
    return splits


def parse_transcript_from_filename(path: str) -> str:
    # transcript is the first underscore-separated part
    # e.g., "But I do not know the schedule for the trial_10-50_..."
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    # Convert transcript to uppercase for correct tokenization
    transcript = parts[0].strip().upper().replace(" ", "▁")
    return transcript


def parse_transcript_original(path: str) -> str:
    """Get original transcript text (not tokenized) for matching"""
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    return parts[0].strip()


def build_transcript_to_number_map(reference_dir: str) -> Dict[str, int]:
    """
    Build a mapping from transcript text to number based on files in reference directory.
    Files are named like: 1.So I think we can do this..wav
    """
    transcript_to_number = {}
    files = glob.glob(os.path.join(reference_dir, "*.wav"))
    
    # Sort by number in filename
    def extract_number(filename):
        basename = os.path.basename(filename)
        try:
            num = int(basename.split('.')[0])
            return num
        except:
            return 999999
    
    files_sorted = sorted(files, key=extract_number)
    
    for filepath in files_sorted:
        basename = os.path.basename(filepath)
        # Extract number and transcript
        parts = basename.split('.', 1)  # Split on first dot
        if len(parts) >= 2:
            number = int(parts[0])
            # Remove .wav extension
            transcript_text = parts[1].replace('.wav', '').strip()
            # Normalize: uppercase, remove extra spaces, and remove trailing dots
            transcript_normalized = transcript_text.upper().strip().rstrip('.')
            transcript_to_number[transcript_normalized] = number
            # Also store without any punctuation for more flexible matching
            transcript_no_punct = ''.join(c for c in transcript_normalized if c.isalnum() or c.isspace())
            if transcript_no_punct != transcript_normalized:
                transcript_to_number[transcript_no_punct] = number
    
    return transcript_to_number


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


def process_one(path: str, split: str, detector: str, out_root: str, text_transform: TextTransform, combine_av: bool, convert_gray: bool = False, file_number: int = None) -> Tuple[bool, str, int, str, str, str]:
    transcript = parse_transcript_from_filename(path)
    print(transcript)

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
    dataset_name = f"video300_{detector}"
    
    # Mouth crop paths
    dst_vid_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_video_seg24s", split)
    dst_txt_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_text_seg24s", split)
    
    # Original with bbox paths
    dst_bbox_dir = os.path.join(out_root, dataset_name, f"{dataset_name}_video_bbox_seg24s", split)

    # Use file_number for naming if provided, otherwise use original filename
    if file_number is not None:
        # Get original transcript text for filename
        transcript_original = parse_transcript_original(path)
        # Create filename like: 1.So I think we can do this.
        output_stem = f"{file_number}.{transcript_original}"
    else:
        rel_name = os.path.basename(path)
        output_stem = os.path.splitext(rel_name)[0]
    
    # Mouth crop files
    dst_vid = os.path.join(dst_vid_dir, output_stem + ".mp4")
    dst_aud = os.path.join(dst_vid_dir, output_stem + ".wav")
    dst_txt = os.path.join(dst_txt_dir, output_stem + ".txt")
    
    # Bbox version files
    dst_bbox_vid = os.path.join(dst_bbox_dir, output_stem + ".mp4")
    dst_bbox_aud = os.path.join(dst_bbox_dir, output_stem + ".wav")

    num_frames = int(video_tensor.shape[0])
    token_id_str = " ".join(map(str, [_.item() for _ in text_transform.tokenize(transcript)]))

    try:
        ensure_dir(dst_vid_dir)
        ensure_dir(dst_txt_dir)
        
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
    parser = argparse.ArgumentParser(description="Preprocess video_300_25p into cropped datasets + bbox videos")
    parser.add_argument("--input_dir", type=str, default="/scratch/th3482/LipVideoData/video_300_25p")
    parser.add_argument("--output_root", type=str, default="/scratch/th3482/LipVideoData/video_300_25p_crops_bbox")
    parser.add_argument("--detectors", type=str, default="mediapipe")
    parser.add_argument("--convert_gray", action="store_true", help="Convert crops to grayscale")
    parser.add_argument("--combine_av", action="store_true", help="Mux audio into mp4 and remove wav")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of videos to process before clearing cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val/test split")
    args = parser.parse_args()

    input_dir = args.input_dir
    out_root = args.output_root
    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    ensure_dir(out_root)
    text_transform = TextTransform()
    video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    print(f"Found {len(video_files)} mp4 files under {input_dir}")

    # Build transcript to number mapping from reference directory
    reference_dir = "/home/th3482/auto-avsr/xtts_project/legal_outputs_terry_1.3x"
    transcript_to_number = build_transcript_to_number_map(reference_dir)
    print(f"Built transcript mapping with {len(transcript_to_number)} entries")

    # Create random split
    split_sizes = {
        "train": 200,
        "val": 50,
        "test": 50
    }
    split_files = create_random_split(video_files, split_sizes, seed=args.seed)
    
    # Save split information for reproducibility
    split_info_dir = os.path.join(out_root, "split_info")
    ensure_dir(split_info_dir)
    for split_name, files in split_files.items():
        with open(os.path.join(split_info_dir, f"{split_name}_files.txt"), "w") as f:
            for file_path in files:
                f.write(f"{os.path.basename(file_path)}\n")

    # Only process val set
    val_files = split_files["val"]
    print(f"\nProcessing only VAL set: {len(val_files)} files")

    for detector in detectors:
        ok, fail = 0, 0
        # Prepare label CSV for val split only
        dataset_name = f"video300_{detector}"
        labels_dir = os.path.join(out_root, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        label_file = os.path.join(labels_dir, f"{dataset_name}_val_transcript_lengths_seg24s.csv")
        # Open file in append mode to preserve progress if killed
        handle = open(label_file, "a")
        print(f"\n=== Processing with detector: {detector} ===")
        
        # Process val files only
        for i, path in enumerate(tqdm(val_files, desc=f"{detector}-val")):
            split = "val"
            # Get file number based on transcript
            transcript_original = parse_transcript_original(path)
            transcript_normalized = transcript_original.upper().strip()
            # Try exact match first
            file_number = transcript_to_number.get(transcript_normalized)
            # If not found, try without punctuation
            if file_number is None:
                transcript_no_punct = ''.join(c for c in transcript_normalized if c.isalnum() or c.isspace())
                file_number = transcript_to_number.get(transcript_no_punct)
            if file_number is None:
                print(f"Warning: Could not find number for transcript: {transcript_original}")
            
            success, split, nframes, token_ids, ds_name, rel_base = process_one(
                path, split, detector, out_root, text_transform, args.combine_av, convert_gray=args.convert_gray, file_number=file_number
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

