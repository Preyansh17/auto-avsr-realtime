#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.checkpoint import (  # noqa: E402
    download_checkpoint,
    extract_state_dict,
    load_online_avsr_module,
    preflight_environment,
    validate_online_state_dict,
)
from online_avsr.data import discover_patient_records  # noqa: E402
from online_avsr.text import (  # noqa: E402
    checkpoint_id,
    compute_wer,
    extract_reference_from_filename,
)
from online_avsr.transforms import AudioTransform, VideoTransform  # noqa: E402


def cut_or_pad(data: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    if data.size(dim) < size:
        padding = size - data.size(dim)
        return torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
    if data.size(dim) > size:
        return data[:size]
    return data


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, payload) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_video_np(path: str) -> Tuple["object", float]:
    import torchvision

    video, _, info = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
    fps = float(info.get("video_fps") or 25.0)
    return video.numpy(), fps


def load_audio(path: str) -> torch.Tensor:
    import torchaudio

    waveform, sample_rate = torchaudio.load(path, normalize=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.transpose(1, 0)


class StreamingPatientAVSR:
    def __init__(self, args, checkpoint_path: str):
        self.args = args
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.checkpoint_id = checkpoint_id(checkpoint_path)
        self.module = load_online_avsr_module(checkpoint_path, args.sp_model_path, self.device)
        self.video_transform = VideoTransform("test")
        self.audio_transform = AudioTransform("test")
        self._init_detector(args.detector)

    def _init_detector(self, detector: str) -> None:
        if detector == "mediapipe":
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            from preparation.detectors.mediapipe.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector()
            self.video_process = VideoProcess(convert_gray=False)
        elif detector == "retinaface":
            from preparation.detectors.retinaface.detector import LandmarksDetector
            from preparation.detectors.retinaface.video_process import VideoProcess

            self.landmarks_detector = LandmarksDetector(device=str(self.device))
            self.video_process = VideoProcess(convert_gray=False)
        else:
            raise ValueError(f"Unknown detector: {detector}")

    def _preprocess_chunk(self, video_np, audio: torch.Tensor, start_frame: int, end_frame: int):
        video_chunk = video_np[start_frame:end_frame]
        audio_start = start_frame * 640
        audio_end = end_frame * 640
        audio_chunk = audio[audio_start:audio_end]

        landmarks = self.landmarks_detector(video_chunk)
        cropped = self.video_process(video_chunk, landmarks)
        if cropped is None:
            raise RuntimeError("No face landmarks detected for chunk")

        video_tensor = torch.tensor(cropped).permute(0, 3, 1, 2)
        video_tensor = self.video_transform(video_tensor)
        audio_tensor = cut_or_pad(audio_chunk, video_tensor.size(0) * 640)
        audio_tensor = self.audio_transform(audio_tensor)

        return audio_tensor.unsqueeze(0).to(self.device), video_tensor.unsqueeze(0).to(self.device)

    def stream_video(self, video_path: str, partials_path: str) -> Dict:
        video_np, fps = load_video_np(video_path)
        audio = load_audio(video_path)
        total_frames = min(len(video_np), len(audio) // 640)
        if total_frames <= 0:
            raise RuntimeError("Video/audio alignment produced zero usable frames")

        state = None
        hypothesis = None
        chunk_records = []
        final_transcript = ""
        start_wall = time.time()

        chunk_frames = int(self.args.chunk_frames)
        chunk_index = 0
        for start_frame in range(0, total_frames, chunk_frames):
            end_frame = min(total_frames, start_frame + chunk_frames)
            if end_frame <= start_frame:
                continue
            chunk_wall = time.time()
            audio_tensor, video_tensor = self._preprocess_chunk(video_np, audio, start_frame, end_frame)
            with torch.no_grad():
                transcript, hypothesis, state = self.module.stream_step(
                    audio_tensor,
                    video_tensor,
                    state=state,
                    hypothesis=hypothesis,
                    beam_width=self.args.beam_width,
                )
            wall_sec = time.time() - chunk_wall
            media_sec = (end_frame - start_frame) / fps
            final_transcript = transcript
            record = {
                "event": "partial",
                "video_path": video_path,
                "chunk_index": chunk_index,
                "chunk_start_sec": start_frame / fps,
                "chunk_end_sec": end_frame / fps,
                "partial_transcript": transcript,
                "wall_sec": wall_sec,
                "media_sec": media_sec,
                "rtf": wall_sec / media_sec if media_sec > 0 else None,
                "detector": self.args.detector,
                "checkpoint_id": self.checkpoint_id,
            }
            append_jsonl(partials_path, record)
            chunk_records.append(record)
            chunk_index += 1

        elapsed = time.time() - start_wall
        media_duration = total_frames / fps
        reference = extract_reference_from_filename(video_path)
        wer = compute_wer(reference, final_transcript) if reference else None
        final_record = {
            "event": "final",
            "video_path": video_path,
            "reference": reference,
            "transcript": final_transcript,
            "wer": wer,
            "chunks": len(chunk_records),
            "total_wall_sec": elapsed,
            "media_sec": media_duration,
            "rtf": elapsed / media_duration if media_duration > 0 else None,
            "detector": self.args.detector,
            "checkpoint_id": self.checkpoint_id,
        }
        append_jsonl(partials_path, final_record)
        return final_record


def resolve_checkpoint(args) -> str:
    if args.checkpoint_url:
        return download_checkpoint(
            args.checkpoint_url,
            os.path.join(PROJECT_ROOT, "cpts", "online_avsr"),
            expected_sha256=args.checkpoint_sha256 or "",
        )
    if args.checkpoint_path:
        return args.checkpoint_path
    if args.model_source == "scratch" and args.run_dir:
        candidates = [
            os.path.join(args.run_dir, "last.ckpt"),
            os.path.join(args.run_dir, "checkpoints", "last.ckpt"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    raise ValueError("Provide --checkpoint-path, --checkpoint-url, or --run-dir with a last.ckpt")


def dry_run(args, checkpoint_path: str) -> None:
    import torch

    env = preflight_environment()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict, _ = extract_state_dict(ckpt)
    validate_online_state_dict(state_dict)
    records = discover_patient_records(args.patient_dir, limit=args.max_videos)
    print(json.dumps(
        {
            "preflight": env,
            "checkpoint_path": checkpoint_path,
            "patient_dir": args.patient_dir,
            "videos_found": len(records),
            "sp_model_path": args.sp_model_path,
        },
        indent=2,
    ))


def parse_args():
    parser = argparse.ArgumentParser(description="True-online patient AVSR inference with Emformer RNN-T.")
    parser.add_argument("--model-source", choices=["pretrained", "scratch"], default="pretrained")
    parser.add_argument("--checkpoint-path", default=os.environ.get("ONLINE_CHECKPOINT"))
    parser.add_argument("--checkpoint-url", default=os.environ.get("CHECKPOINT_URL"))
    parser.add_argument("--checkpoint-sha256", default=os.environ.get("CHECKPOINT_SHA256"))
    parser.add_argument("--run-dir", default=os.environ.get("RUN_DIR"))
    parser.add_argument("--sp-model-path", required=True)
    parser.add_argument("--patient-dir", default=os.environ.get("PATIENT_DIR", "/scratch/th3482/LipVideoData/patient_25p"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs", "online_avsr")))
    parser.add_argument("--detector", choices=["mediapipe", "retinaface"], default=os.environ.get("DETECTOR", "mediapipe"))
    parser.add_argument("--chunk-frames", type=int, default=int(os.environ.get("CHUNK_FRAMES", "64")))
    parser.add_argument("--beam-width", type=int, default=int(os.environ.get("BEAM_WIDTH", "20")))
    parser.add_argument("--max-videos", type=int, default=int(os.environ.get("MAX_VIDEOS", "0")) or None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dry_run:
        dry_run(args, checkpoint_path)
        return

    env = preflight_environment()
    run_dir = os.path.join(args.output_dir, f"online_patient_av_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)
    partials_path = os.path.join(run_dir, "partials.jsonl")
    finals_path = os.path.join(run_dir, "final_transcripts.json")
    summary_path = os.path.join(run_dir, "summary.txt")
    write_json(os.path.join(run_dir, "preflight.json"), env)

    records = discover_patient_records(args.patient_dir, limit=args.max_videos)
    if not records:
        raise FileNotFoundError(f"No patient videos found in {args.patient_dir}")

    pipeline = StreamingPatientAVSR(args, checkpoint_path)
    finals: List[Dict] = []
    for record in records:
        try:
            finals.append(pipeline.stream_video(record.path, partials_path))
        except Exception as exc:
            error_record = {
                "event": "error",
                "video_path": record.path,
                "error": str(exc),
                "detector": args.detector,
                "checkpoint_id": pipeline.checkpoint_id,
            }
            append_jsonl(partials_path, error_record)
            finals.append(error_record)

    write_json(finals_path, finals)
    successful = [item for item in finals if item.get("event") == "final"]
    avg_wer = None
    avg_rtf = None
    if successful:
        wers = [item["wer"] for item in successful if item.get("wer") is not None]
        rtfs = [item["rtf"] for item in successful if item.get("rtf") is not None]
        avg_wer = sum(wers) / len(wers) if wers else None
        avg_rtf = sum(rtfs) / len(rtfs) if rtfs else None
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Online Patient AVSR Summary\n")
        f.write(f"checkpoint_path: {checkpoint_path}\n")
        f.write(f"checkpoint_id: {pipeline.checkpoint_id}\n")
        f.write(f"patient_dir: {args.patient_dir}\n")
        f.write(f"detector: {args.detector}\n")
        f.write(f"videos_total: {len(records)}\n")
        f.write(f"videos_successful: {len(successful)}\n")
        f.write(f"avg_wer: {avg_wer}\n")
        f.write(f"avg_rtf: {avg_rtf}\n")
        f.write(f"partials_jsonl: {partials_path}\n")
        f.write(f"final_transcripts_json: {finals_path}\n")
    print(f"Completed online patient AVSR. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
