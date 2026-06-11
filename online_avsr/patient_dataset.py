"""Patient AV dataset for the streaming recipe.

Reads the auto-avsr label CSV layout ("dataset_name,rel_path,input_length,
token ids") with token ids re-generated for the 1023-piece SentencePiece
model (scripts/regenerate_patient_labels.py), resolving media as
root_dir/dataset_name/rel_path (25 fps mouth-ROI mp4 + sibling 16 kHz wav).
CSV token ids are the canonical targets; text decoded from them is kept for
evaluation. Directory scanning (filename-derived text) remains available
for unlabeled inference.
"""

import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .text import extract_reference_from_filename


@dataclass
class PatientSampleRecord:
    path: str
    text: str
    frames: int
    token_ids: List[int] = field(default_factory=list)


def cut_or_pad(data: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    if data.size(dim) < size:
        return torch.nn.functional.pad(data, (0, 0, 0, size - data.size(dim)), "constant")
    if data.size(dim) > size:
        return data[:size]
    return data


def read_label_csv(root_dir: str, label_file: str, sp_model=None) -> List[PatientSampleRecord]:
    records = []
    with open(label_file, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            dataset_name, rel_path, frame_count, token_str = parts[0], parts[1], parts[2], parts[3]
            token_ids = [int(t) for t in token_str.split()] if token_str.strip() else []
            text = sp_model.decode(token_ids) if (sp_model and token_ids) else extract_reference_from_filename(rel_path)
            records.append(
                PatientSampleRecord(
                    path=os.path.join(root_dir, dataset_name, rel_path),
                    text=text,
                    frames=int(frame_count),
                    token_ids=token_ids,
                )
            )
    return records


def discover_patient_records(
    root_dir: str, label_file: Optional[str] = None, limit: Optional[int] = None, sp_model=None
) -> List[PatientSampleRecord]:
    if label_file:
        records = read_label_csv(root_dir, label_file, sp_model=sp_model)
    else:
        video_paths = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            video_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        video_paths.sort()
        records = [
            PatientSampleRecord(path=path, text=extract_reference_from_filename(path), frames=0)
            for path in video_paths
        ]
    if limit and limit > 0:
        records = records[:limit]
    return records


def load_video(path: str) -> torch.Tensor:
    import torchvision

    video = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    return video.permute(0, 3, 1, 2)  # T, C, H, W


def load_audio_for_video(path: str) -> torch.Tensor:
    import torchaudio

    audio_path = os.path.splitext(path)[0] + ".wav"
    source = audio_path if os.path.isfile(audio_path) else path
    waveform, sample_rate = torchaudio.load(source, normalize=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return torch.mean(waveform, dim=0, keepdim=True).transpose(1, 0)  # N, 1


class PatientAVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        label_file: Optional[str] = None,
        sp_model=None,
        limit: Optional[int] = None,
        max_frames: Optional[int] = 600,
    ):
        records = discover_patient_records(root_dir, label_file=label_file, limit=None, sp_model=sp_model)
        if max_frames:
            kept = [r for r in records if not r.frames or r.frames <= max_frames]
            if len(kept) < len(records):
                print(f"PatientAVDataset: dropped {len(records) - len(kept)} clips over {max_frames} frames")
            records = kept
        if limit and limit > 0:
            records = records[:limit]
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        video = load_video(record.path)
        audio = load_audio_for_video(record.path)
        audio = cut_or_pad(audio, len(video) * 640)
        return {
            "path": record.path,
            "text": record.text,
            "token_ids": record.token_ids,
            "video": video,
            "audio": audio,
        }
