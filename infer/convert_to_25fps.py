#!/usr/bin/env python3
"""
Convert all videos in specified directories to 25fps.
Output videos are saved to corresponding *_25p directories.
"""
import os
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def convert_video_to_25fps(input_path: str, output_path: str) -> bool:
    """
    Convert a single video to 25fps using ffmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-r", "25",  # Force 25 fps
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout per video
        )
        return True
    except subprocess.CalledProcessError as e:
        try:
            err = e.stderr.decode("utf-8", errors="ignore")
        except Exception:
            err = str(e)
        print(f"Failed to convert {input_path}: {err}")
        return False
    except subprocess.TimeoutExpired:
        print(f"Timeout converting {input_path}")
        return False
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def process_directory(input_dir: str, output_dir: str, extensions=None) -> dict:
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        extensions: List of video file extensions to process
        
    Returns:
        Dictionary with processing statistics
    """
    if extensions is None:
        extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    
    # Collect all video files
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    video_files.sort()
    
    if len(video_files) == 0:
        print(f"No videos found in {input_dir}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    # Create output directory
    ensure_dir(output_dir)
    
    print(f"\nProcessing {len(video_files)} videos from {input_dir}")
    print(f"Output directory: {output_dir}")
    
    stats = {"total": len(video_files), "success": 0, "failed": 0, "skipped": 0}
    
    pbar = tqdm(video_files, desc=f"Converting {os.path.basename(input_dir)}")
    
    for video_file in pbar:
        basename = os.path.basename(video_file)
        output_file = os.path.join(output_dir, basename)
        
        # Skip if output already exists
        if os.path.exists(output_file):
            stats["skipped"] += 1
            pbar.set_postfix({"Success": stats["success"], "Failed": stats["failed"], "Skipped": stats["skipped"]})
            continue
        
        # Convert video
        success = convert_video_to_25fps(video_file, output_file)
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        pbar.set_postfix({"Success": stats["success"], "Failed": stats["failed"], "Skipped": stats["skipped"]})
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to 25fps and save to *_25p directories"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/scratch/th3482/LipVideoData",
        help="Root directory containing video folders"
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        default=["patient", "video_50", "video_300"],
        help="List of subdirectories to process"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video 25fps Conversion Tool")
    print("=" * 60)
    
    total_stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    for subdir in args.dirs:
        input_dir = os.path.join(args.root, subdir)
        output_dir = os.path.join(args.root, f"{subdir}_25p")
        
        if not os.path.isdir(input_dir):
            print(f"Warning: Input directory does not exist: {input_dir}")
            continue
        
        stats = process_directory(input_dir, output_dir)
        
        # Accumulate statistics
        for key in total_stats:
            total_stats[key] += stats[key]
        
        print(f"\n{subdir} Statistics:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")
    
    print("\n" + "=" * 60)
    print("Overall Statistics:")
    print(f"  Total videos: {total_stats['total']}")
    print(f"  Successfully converted: {total_stats['success']}")
    print(f"  Failed: {total_stats['failed']}")
    print(f"  Skipped (already exist): {total_stats['skipped']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
