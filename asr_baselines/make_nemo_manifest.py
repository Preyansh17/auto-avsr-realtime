#!/usr/bin/env python3
"""Build NeMo ASR manifests (jsonl) from the patient label CSVs.

Each line: {"audio_filepath": <wav>, "duration": <sec>, "text": <transcript>}.
Text is decoded from the canonical SentencePiece token ids (same targets the AV
Emformer + Whisper use), so all three models train/eval on identical references.

  python -m asr_baselines.make_nemo_manifest \
    --root-dir $ROOT --label-file train.csv --out $EXP/train_manifest.jsonl
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.patient_audio import TARGET_SR, load_audio_examples, load_waveform  # noqa: E402

DEFAULT_SPM = os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model")


def audio_path_and_duration(media_path):
    """Resolve the sibling wav (or the mp4 itself) and its duration in seconds.
    Uses torchaudio.info when possible (cheap), else decodes."""
    import torchaudio

    wav = os.path.splitext(media_path)[0] + ".wav"
    src = wav if os.path.isfile(wav) else media_path
    try:
        info = torchaudio.info(src)
        dur = info.num_frames / info.sample_rate
        if info.sample_rate != TARGET_SR:
            dur = dur  # duration is sample-rate invariant
        return src, float(dur)
    except Exception:
        w = load_waveform(media_path)
        return src, float(w.numel()) / TARGET_SR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", required=True)
    p.add_argument("--label-file", required=True)
    p.add_argument("--sp-model-path", default=DEFAULT_SPM)
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--lowercase", action="store_true", default=True)
    p.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    args = p.parse_args()

    examples = load_audio_examples(args.root_dir, args.label_file, args.sp_model_path,
                                   max_frames=args.max_frames)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n, total_dur = 0, 0.0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            src, dur = audio_path_and_duration(ex.path)
            text = ex.text.lower() if args.lowercase else ex.text
            f.write(json.dumps({"audio_filepath": src, "duration": dur, "text": text}) + "\n")
            n += 1
            total_dur += dur
    print(f"wrote {n} entries ({total_dur/60:.1f} min) -> {args.out}")


if __name__ == "__main__":
    main()
