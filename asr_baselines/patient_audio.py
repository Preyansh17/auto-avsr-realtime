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
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    # (word, end_time_sec) per spoken word, from a forced alignment (MFA
    # TextGrid). Only populated when load_audio_examples is given aligned_csv;
    # None means no alignment (the example trains untruncated). Used by
    # whisper_finetune's train-time buffer truncation to build a valid prefix
    # target when the input audio is cut to a partial buffer.
    words: Optional[List[Tuple[str, float]]] = None


# The "words" IntervalTier of a long-format MFA TextGrid: every interval is
# (xmin, xmax, text); empty text == silence. We keep only spoken words and
# their end time (xmax).
_TG_INTERVAL = re.compile(
    r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"'
)


def parse_textgrid_words(path: str) -> Optional[List[Tuple[str, float]]]:
    """Return [(word, end_sec), ...] for the 'words' tier, or None if unparseable."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'name\s*=\s*"words".*?(?=\n\s*item\s*\[|\Z)', content, re.DOTALL)
    if not m:
        return None
    words = []
    for im in _TG_INTERVAL.finditer(m.group(0)):
        text = im.group(3).strip()
        if text:
            words.append((text, float(im.group(2))))
    return words or None


def _load_alignment_index(aligned_csv: str) -> dict:
    """Map wav basename -> TextGrid path from the CarelessWhisper joined CSV
    (TAB-separated wav_path\ttg_path\traw_text, CRLF line endings, filenames
    may contain spaces so split on TAB only)."""
    index = {}
    with open(aligned_csv, encoding="utf-8") as f:
        header = f.readline()  # wav_path\ttg_path\traw_text
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 2:
                continue
            index[os.path.basename(parts[0])] = parts[1]
    return index


def load_audio_examples(
    root_dir: str,
    label_file: str,
    sp_model_path: str,
    max_frames: Optional[int] = None,
    limit: Optional[int] = None,
    aligned_csv: Optional[str] = None,
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
    if aligned_csv:
        _attach_alignments(out, aligned_csv)
    return out


def _attach_alignments(examples: List[AudioExample], aligned_csv: str) -> None:
    """Populate .words on each example whose forced alignment (a) exists and
    (b) matches its target text word-for-word. Mismatches/misses stay None
    (that example trains untruncated). Logs coverage."""
    index = _load_alignment_index(aligned_csv)
    matched = mismatched = missing = 0
    shown = 0
    for ex in examples:
        wav_base = os.path.basename(os.path.splitext(ex.path)[0] + ".wav")
        tg_path = index.get(wav_base)
        if not tg_path or not os.path.isfile(tg_path):
            missing += 1
            continue
        words = parse_textgrid_words(tg_path)
        if not words:
            missing += 1
            continue
        # Prefix truncation is only valid if the aligned word sequence IS the
        # target word sequence -- otherwise a k-word prefix isn't a prefix of
        # the training target. Keep only exact matches.
        aligned = [w.lower() for w, _ in words]
        target = ex.text.lower().split()
        if aligned == target:
            ex.words = words
            matched += 1
        else:
            mismatched += 1
            if shown < 3:
                print(f"[align mismatch] {wav_base}\n  aligned={aligned}\n  target={target}")
                shown += 1
    print(f"alignments: matched {matched}/{len(examples)} "
          f"(mismatched {mismatched}, missing {missing})")


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
