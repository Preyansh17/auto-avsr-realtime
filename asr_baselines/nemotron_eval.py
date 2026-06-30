#!/usr/bin/env python3
"""Evaluate a finetuned (or pretrained) Nemotron model on a NeMo manifest.

Reports WER/CER (shared normalizer, comparable to Whisper + Emformer) and a
rough RTF / per-utterance latency. Pass a .nemo path or the hub id for the
zero-shot baseline.

  python -m asr_baselines.nemotron_eval --model $EXP/nemotron_patient.nemo \
    --manifest val.jsonl --out hyps.tsv

NOTE: this uses offline transcribe(). True cache-aware streaming WER+latency
(chunked decode with carried state) is a follow-up using NeMo's
cache_aware_streaming utils; the model is the same, only the decode loop differs.
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.metrics import compute_wer_cer  # noqa: E402


def read_manifest(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import nemo.collections.asr as nemo_asr

    if args.model.endswith(".nemo") and os.path.isfile(args.model):
        model = nemo_asr.models.ASRModel.restore_from(args.model)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()

    items = read_manifest(args.manifest)
    paths = [it["audio_filepath"] for it in items]
    refs = [it["text"] for it in items]
    total_audio = sum(it.get("duration", 0.0) for it in items)

    t0 = time.time()
    out = model.transcribe(paths, batch_size=args.batch_size)
    elapsed = time.time() - t0
    # NeMo transcribe returns either list[str] or list[Hypothesis] (or a tuple
    # for transducer models); normalize to text.
    if isinstance(out, tuple):
        out = out[0]
    hyps = [h.text if hasattr(h, "text") else h for h in out]

    m = compute_wer_cer(refs, hyps)
    rtf = elapsed / total_audio if total_audio else float("nan")
    print(f"N={len(refs)}  WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"RTF={rtf:.3f}  ({elapsed:.1f}s / {total_audio:.1f}s audio, "
          f"{1000*elapsed/max(len(refs),1):.0f} ms/utt)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("path\tref\thyp\n")
            for path, ref, hyp in zip(paths, refs, hyps):
                f.write(f"{path}\t{ref}\t{hyp}\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
