"""Lightning data module for patient AV fine-tuning of the streaming model."""

from typing import Optional

import torch

from .module import AVBatch
from .patient_dataset import PatientAVDataset
from .transforms import AudioTransform, VideoTransform

try:
    from pytorch_lightning import LightningDataModule
except Exception:  # pragma: no cover - preflight catches this in cluster runs.
    LightningDataModule = object


class PatientAVCollate:
    """Applies transforms and pads a batch. Targets come from the label CSV
    token ids (1023-piece model) when present; text is encoded as a fallback
    for directory-scanned datasets.

    frame_size: 88 for the recipe architecture, 44 for the device model.
    """

    def __init__(self, sp_model, subset: str, frame_size: int = 88, specaug: bool = False):
        self.sp_model = sp_model
        train = subset == "train"
        # specaug only affects the train pipeline (val/test stay deterministic).
        self.video_transform = VideoTransform(
            "train" if train else "test", frame_size=frame_size, specaug=specaug and train
        )
        self.audio_transform = AudioTransform("train" if train else "test", specaug=specaug and train)

    def _target(self, sample):
        token_ids = sample.get("token_ids") or self.sp_model.encode(sample["text"].lower())
        if not token_ids:
            token_ids = [self.sp_model.unk_id()]
        return torch.tensor(token_ids, dtype=torch.int32)

    def __call__(self, samples):
        videos, audios, targets = [], [], []
        for sample in samples:
            video, audio = sample["video"], sample["audio"]
            length = min(len(video), len(audio) // 640)
            if length <= 0:
                continue
            videos.append(self.video_transform(video[:length].float()))
            audios.append(self.audio_transform(audio[: length * 640]))
            targets.append(self._target(sample))

        if not videos:
            raise ValueError("No usable samples in batch")

        video_lengths = torch.tensor([v.shape[0] for v in videos], dtype=torch.int32)
        audio_lengths = torch.tensor([a.shape[0] // 640 for a in audios], dtype=torch.int32)
        target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.int32)
        padded_videos = torch.nn.utils.rnn.pad_sequence(videos, batch_first=True)
        padded_audios = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(
            dtype=torch.int32
        )
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
        max_frames: Optional[int] = 600,
        frame_size: int = 88,
        specaug: bool = False,
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
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.specaug = specaug

    def _loader(self, label_file, subset, shuffle=False):
        ds = PatientAVDataset(
            self.root_dir,
            label_file=label_file,
            sp_model=self.sp_model,
            limit=self.limit,
            max_frames=self.max_frames,
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=PatientAVCollate(
                self.sp_model, subset, frame_size=self.frame_size, specaug=self.specaug
            ),
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_file, "train", shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_file, "val", shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_file, "test", shuffle=False)
