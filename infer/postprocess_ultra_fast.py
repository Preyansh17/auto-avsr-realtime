#!/usr/bin/env python3
import os
import re
import json
import math
import subprocess
from typing import List, Tuple
import tempfile


def normalize_text(text: str) -> List[str]:
    if text is None:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference)
    hyp_tokens = normalize_text(hypothesis)
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else 1.0
    dist = levenshtein_distance(ref_tokens, hyp_tokens)
    return dist / float(len(ref_tokens))


def extract_groundtruth_from_filename(video_path: str) -> str:
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    # ground truth is the first segment before the first underscore
    # e.g., "It is about the exposure in this case_47-50_repeat5_..."
    gt = name.split("_")[0]
    # Replace dashes with spaces for cleaner tokenization
    gt = gt.replace("-", " ")
    return gt


def extract_repeat_from_filename(video_path: str) -> str:
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    parts = name.split("_")
    for part in parts:
        if re.fullmatch(r"repeat\d+", part.lower()):
            return part
    # fallback: try to find pattern anywhere
    m = re.search(r"repeat\d+", name.lower())
    if m:
        return m.group(0)
    # Patient data may not have repeat token; return empty string
    return ""


def parse_json_identity(json_path: str) -> Tuple[str, str, str]:
    base = os.path.basename(json_path)
    name, _ = os.path.splitext(base)
    # Patterns:
    #  video: ultra_fast_{dir_tag}_{detector}_{ts}
    #  av:    ultra_fast_av_{dir_tag}_{detector}_{ts}
    #  audio: ultra_fast_audio_{dir_tag}_{ts}
    #  patient_video: ultra_fast_patient_video_{detector}_{ts}
    #  patient_audio: ultra_fast_patient_audio_{ts}
    #  patient_av: ultra_fast_patient_av_{detector}_{ts}
    if name.startswith("ultra_fast_patient_av_"):
        modality = "av"
        rest = name[len("ultra_fast_patient_av_"):]
        parts = rest.split("_")
        if len(parts) < 2:
            return (modality, "patient", "unknown")
        detector = parts[-2]
        return (modality, "patient", detector)
    elif name.startswith("ultra_fast_patient_audio_"):
        modality = "audio"
        rest = name[len("ultra_fast_patient_audio_"):]
        return (modality, "patient", "none")
    elif name.startswith("ultra_fast_patient_video_"):
        modality = "video"
        rest = name[len("ultra_fast_patient_video_"):]
        parts = rest.split("_")
        if len(parts) < 2:
            return (modality, "patient", "unknown")
        detector = parts[-2]
        return (modality, "patient", detector)
    elif name.startswith("ultra_fast_av_"):
        modality = "av"
        rest = name[len("ultra_fast_av_"):]
    elif name.startswith("ultra_fast_audio_"):
        modality = "audio"
        rest = name[len("ultra_fast_audio_"):]
        # audio has no detector, format: dir_tag_ts
        parts = rest.split("_")
        if len(parts) < 2:
            return (modality, "unknown", "none")
        dir_tag = "_".join(parts[:-1])
        return (modality, dir_tag, "none")
    elif name.startswith("ultra_fast_"):
        modality = "video"
        rest = name[len("ultra_fast_"):]
    else:
        modality = "video"
        rest = name

    parts = rest.split("_")
    if len(parts) < 3:
        # Fallback: unknown pattern
        return (modality, "unknown", "unknown")
    # last two are detector and ts; dir_tag is the remaining prefix (may contain underscores)
    detector = parts[-2]
    dir_tag = "_".join(parts[:-2])
    return (modality, dir_tag, detector)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def overlay_transcript_ffmpeg(input_video: str, transcript: str, output_video: str) -> None:
    # Reliable approach: draw a black bar (drawbox) and overlay outlined text (drawtext), enforce yuv420p and +faststart
    text = transcript if transcript is not None else ""
    safe_text = (
        str(text)
        .replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", r"\\'")
        .replace("\n", r"\\n")
        .replace("\r", "")
    )
    candidate_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    fontfile = next((p for p in candidate_fonts if os.path.isfile(p)), None)
    font_part = f"fontfile={fontfile}:" if fontfile else ""

    # drawtext only, large font, with border for readability; no background bar
    filter_chain = (
        f"drawtext={font_part}text='{safe_text}':fontcolor=white:fontsize=48:"
        "bordercolor=black:borderw=4:"
        "x=(w-text_w)/2:y=h-(text_h*2)-30"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", filter_chain,
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "copy",
        output_video,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        try:
            err = e.stderr.decode("utf-8", errors="ignore")
        except Exception:
            err = str(e)
        print("ffmpeg overlay failed:\n" + err)
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess ultra-fast inference outputs: compute WER, overlay transcripts, and export videos.")
    parser.add_argument("json_path", type=str, help="Path to results JSON file, e.g., /home/th3482/auto-avsr/ultra_fast_video_50_mediapipe_123.json")
    parser.add_argument("--root_input_dir", type=str, default="/scratch/th3482/LipVideoData/25p_results", help="Root dir that contains the original video dirs")
    args = parser.parse_args()

    json_path = args.json_path
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    modality, dir_tag_from_name, detector = parse_json_identity(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total_wer = 0.0
    total_time = 0.0
    count = 0

    # Construct output directory
    if detector == "none":
        out_dir = os.path.join(args.root_input_dir, f"{dir_tag_from_name}_{modality}")
    else:
        out_dir = os.path.join(args.root_input_dir, f"{dir_tag_from_name}_{detector}_{modality}")
    ensure_dir(out_dir)

    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "success":
            continue
        # Support both video_path (video/av) and audio_path (audio)
        vpath = item.get("video_path") or item.get("audio_path")
        transcript = item.get("transcript")
        if not vpath or not transcript:
            continue

        gt = extract_groundtruth_from_filename(vpath)
        wer = compute_wer(gt, str(transcript))
        ptime = float(item.get("processing_time", 0.0))
        total_wer += wer
        total_time += ptime
        count += 1

        # Export overlaid video
        in_exists = os.path.isfile(vpath)
        if not in_exists:
            # Try to reconstruct from dir_tag and filename
            alt_path = os.path.join(args.root_input_dir, dir_tag_from_name, os.path.basename(vpath))
            if os.path.isfile(alt_path):
                vpath = alt_path
            else:
                print(f"Skip overlay, source not found: {vpath}")
                continue

        def sanitize_filename_component(text: str, max_len: int = 80) -> str:
            text = text.strip()
            text = re.sub(r"\s+", " ", text)
            # keep alnum, space, dash, underscore, dot; keep spaces (no underscore join)
            text = re.sub(r"[^A-Za-z0-9 \-\._]+", "", text)
            if len(text) > max_len:
                text = text[:max_len]
            return text

        gt_comp = sanitize_filename_component(gt, 60)
        pred_comp = sanitize_filename_component(str(transcript), 60)

        repeat_raw = extract_repeat_from_filename(vpath)
        repeat_comp = sanitize_filename_component(repeat_raw, 20)

        # New naming: gt_repeat_wer_pred (WER placed before predicted transcript)
        # For patient data without repeat, omit the repeat component
        if repeat_comp:
            out_name = f"{gt_comp}_{repeat_comp}_wer{wer:.3f}_{pred_comp}.mp4"
        else:
            out_name = f"{gt_comp}_wer{wer:.3f}_{pred_comp}.mp4"
        out_path = os.path.join(out_dir, out_name)
        try:
            overlay_transcript_ffmpeg(vpath, str(transcript), out_path)
            print(f"Saved with transcript: {out_path}")
        except Exception as e:
            print(f"Overlay failed for {vpath}: {e}")

    avg_wer = (total_wer / count) if count > 0 else math.nan
    avg_time = (total_time / count) if count > 0 else math.nan

    print("\n=== Summary ===")
    print(f"JSON: {json_path}")
    print(f"Modality: {modality} | Detector: {detector} | Dir tag: {dir_tag_from_name}")
    print(f"Output dir: {out_dir}")
    print(f"Num successful: {count}")
    print(f"Average WER: {avg_wer:.4f}" if not math.isnan(avg_wer) else "Average WER: N/A")
    print(f"Average processing time (s): {avg_time:.3f}" if not math.isnan(avg_time) else "Average processing time: N/A")


if __name__ == "__main__":
    main()


