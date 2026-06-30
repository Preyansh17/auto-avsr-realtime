"""(waveform, text) dataset over the patient label CSVs, for the audio-only
ASR baselines (Whisper, Nemotron).

Reuses the AV repo's CSV reader: each row is
"dataset_name,rel_path,frames,token_ids", media at
root_dir/dataset_name/rel_path. Text is decoded from the SentencePiece token
ids (the canonical labels) -- the SAME targets the AV Emformer trains on, so
WERs are comparable. Video is ignored; only the sibling 16 kHz .wav (or the
mp4's audio track) is loaded.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch
import torchaudio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.patient_dataset import read_label_csv  # noqa: E402
from online_avsr.text import load_sentencepiece_model  # noqa: E402

TARGET_SR = 16000


@dataclass
class AudioExample:
    path: str
    text: str
    frames: int


def load_audio_examples(
    root_dir: str,
    label_file: str,
    sp_model_path: str,
    max_frames: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[AudioExample]:
    sp_model = load_sentencepiece_model(sp_model_path)
    records = read_label_csv(root_dir, label_file, sp_model=sp_model)
    out = []
    for r in records:
        if not r.text:
            continue
        if max_frames and r.frames and r.frames > max_frames:
            continue
        out.append(AudioExample(path=r.path, text=r.text, frames=r.frames))
    if limit and limit > 0:
        out = out[:limit]
    return out


def load_waveform(path: str) -> torch.Tensor:
    """Mono 16 kHz float waveform, shape (num_samples,)."""
    audio_path = os.path.splitext(path)[0] + ".wav"
    source = audio_path if os.path.isfile(audio_path) else path
    wav, sr = torchaudio.load(source, normalize=True)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)


class PatientAudioDataset(torch.utils.data.Dataset):
    """Generic (waveform, text) dataset. The Whisper/Nemotron harnesses wrap
    this with their own feature extractor / tokenizer collators."""

    def __init__(
        self,
        root_dir: str,
        label_file: str,
        sp_model_path: str,
        max_frames: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        self.examples = load_audio_examples(
            root_dir, label_file, sp_model_path, max_frames=max_frames, limit=limit
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "path": ex.path,
            "text": ex.text,
            "audio": load_waveform(ex.path),
            "sampling_rate": TARGET_SR,
        }
