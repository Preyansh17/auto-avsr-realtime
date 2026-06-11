import csv
import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import torch

from .module import AVBatch
from .text import extract_reference_from_filename
from .transforms import AudioTransform, VideoTransform

try:
    from pytorch_lightning import LightningDataModule
except Exception:  # pragma: no cover - preflight catches this in cluster runs.
    LightningDataModule = object


@dataclass
class PatientSampleRecord:
    path: str
    text: str
    frames: int


def _read_patient_csv(root_dir: str, label_file: str) -> List[PatientSampleRecord]:
    records = []
    with open(label_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            dataset_name, rel_path, frame_count = row[:3]
            path = os.path.join(root_dir, dataset_name, rel_path)
            records.append(
                PatientSampleRecord(
                    path=path,
                    text=extract_reference_from_filename(rel_path),
                    frames=int(frame_count),
                )
            )
    return records


def discover_patient_records(root_dir: str, label_file: Optional[str] = None, limit: Optional[int] = None) -> List[PatientSampleRecord]:
    if label_file:
        records = _read_patient_csv(root_dir, label_file)
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


def load_video(path: str):
    import torchvision

    video = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    return video.permute(0, 3, 1, 2)


def load_audio_for_video(path: str):
    import torchaudio

    audio_path = os.path.splitext(path)[0] + ".wav"
    source = audio_path if os.path.isfile(audio_path) else path
    waveform, sample_rate = torchaudio.load(source, normalize=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.transpose(1, 0)


class PatientAVOnlineDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, label_file: Optional[str] = None, limit: Optional[int] = None):
        self.records = discover_patient_records(root_dir, label_file=label_file, limit=limit)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "path": record.path,
            "text": record.text,
            "video": load_video(record.path),
            "audio": load_audio_for_video(record.path),
        }


class PatientAVCollate:
    def __init__(self, sp_model, subset: str):
        self.sp_model = sp_model
        self.video_transform = VideoTransform("train" if subset == "train" else "test")
        self.audio_transform = AudioTransform("train" if subset == "train" else "test")

    def _target(self, text: str):
        token_ids = self.sp_model.encode(text.lower())
        if not token_ids:
            token_ids = [self.sp_model.unk_id()]
        return torch.tensor(token_ids, dtype=torch.int32)

    def __call__(self, samples):
        videos = []
        audios = []
        targets = []
        for sample in samples:
            video = sample["video"]
            audio = sample["audio"]
            length = min(len(video), len(audio) // 640)
            if length <= 0:
                continue
            video = video[:length]
            audio = audio[: length * 640]
            videos.append(self.video_transform(video))
            audios.append(self.audio_transform(audio))
            targets.append(self._target(sample["text"]))

        if not videos:
            raise ValueError("No usable samples in batch")

        video_lengths = torch.tensor([v.shape[0] for v in videos], dtype=torch.int32)
        audio_lengths = torch.tensor([a.shape[0] // 640 for a in audios], dtype=torch.int32)
        target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.int32)
        padded_videos = torch.nn.utils.rnn.pad_sequence(videos, batch_first=True)
        padded_audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(dtype=torch.int32)
        return AVBatch(padded_audios, padded_videos, audio_lengths, video_lengths, padded_targets, target_lengths)


class PatientAVDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        sp_model,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.sp_model = sp_model
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file or val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit

    def _loader(self, label_file, subset, shuffle=False):
        ds = PatientAVOnlineDataset(self.root_dir, label_file=label_file, limit=self.limit)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=PatientAVCollate(self.sp_model, subset),
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_file, "train", shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_file, "val", shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_file, "test", shuffle=False)
