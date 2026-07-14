#!/usr/bin/env python3
"""Build CarelessWhisper finetune inputs from the patient label CSVs.

CarelessWhisper's training (training_code/train.py in their checkout) needs a
CSV with columns wav_path, tg_path, raw_text, where tg_path is a Montreal
Forced Aligner .TextGrid (word alignments drive the streaming loss: the model
learns which words are decidable from a causal prefix). Two steps:

1. `corpus`: lay out an MFA input corpus -- one <utt>.wav symlink + <utt>.lab
   transcript per clip (utt name = label-CSV rel path, "/" -> "__", so names
   are unique and reversible). Then align on the cluster:

     mfa align --clean <corpus_dir> english_us_arpa english_us_arpa <aligned_dir>

   (slurm/carelesswhisper_mfa_align.sbatch wraps this.)

2. `csv`: join the label CSV against <aligned_dir>/<utt>.TextGrid into the
   training CSV. MFA silently skips clips it can't align (OOV-heavy or
   audio-quality failures) -- those rows are DROPPED and counted; check the
   drop count before trusting a training run, a big drop on dysarthric speech
   would itself be a finding.

Text comes from the canonical SentencePiece ids via patient_audio, same as
every other baseline, so WER stays comparable.

  python -m asr_baselines.make_carelesswhisper_dataset corpus \
    --root-dir $DATA --label-file $LABELS/train_spm1023.csv \
    --sp-model spm/unigram/unigram1023_units.txt.model --out-dir $CORPUS/train

  python -m asr_baselines.make_carelesswhisper_dataset csv \
    --root-dir $DATA --label-file $LABELS/train_spm1023.csv \
    --sp-model spm/unigram/unigram1023_units.txt.model \
    --aligned-dir $ALIGNED/train --out-csv $OUT/train.csv
"""

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from asr_baselines.patient_audio import load_audio_examples  # noqa: E402


def utt_name(path, root_dir):
    rel = os.path.relpath(os.path.splitext(path)[0], root_dir)
    return rel.replace(os.sep, "__")


def wav_for(path):
    """Label CSV paths point at video-only lip-crop mp4s; audio is the sibling
    16kHz .wav (same convention as patient_audio.load_waveform)."""
    wav = os.path.splitext(path)[0] + ".wav"
    return wav if os.path.isfile(wav) else path


def cmd_corpus(args, examples):
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for ex in examples:
        wav = wav_for(ex.path)
        name = utt_name(ex.path, args.root_dir)
        link = os.path.join(args.out_dir, name + ".wav")
        if not os.path.exists(link):
            os.symlink(os.path.abspath(wav), link)
        with open(os.path.join(args.out_dir, name + ".lab"), "w", encoding="utf-8") as f:
            f.write(ex.text.strip() + "\n")
        n += 1
    print(f"corpus: {n} utterances -> {args.out_dir}")


def cmd_csv(args, examples):
    rows, dropped = [], []
    for ex in examples:
        name = utt_name(ex.path, args.root_dir)
        tg = os.path.join(args.aligned_dir, name + ".TextGrid")
        if os.path.isfile(tg):
            rows.append((os.path.abspath(wav_for(ex.path)), os.path.abspath(tg), ex.text.strip()))
        else:
            dropped.append(name)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)  # quoted commas: their loader is pandas.read_csv
        w.writerow(["wav_path", "tg_path", "raw_text"])
        w.writerows(rows)
    print(f"csv: {len(rows)} rows -> {args.out_csv}")
    if dropped:
        print(f"[warn] {len(dropped)}/{len(examples)} clips had no TextGrid (MFA "
              f"failed/skipped them) and were DROPPED:")
        for name in dropped:
            print(f"  {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("corpus", "csv"):
        s = sub.add_parser(name)
        s.add_argument("--root-dir", required=True)
        s.add_argument("--label-file", required=True)
        s.add_argument("--sp-model", required=True)
        if name == "corpus":
            s.add_argument("--out-dir", required=True)
        else:
            s.add_argument("--aligned-dir", required=True)
            s.add_argument("--out-csv", required=True)
    args = p.parse_args()

    examples = load_audio_examples(args.root_dir, args.label_file, args.sp_model)
    if args.cmd == "corpus":
        cmd_corpus(args, examples)
    else:
        cmd_csv(args, examples)


if __name__ == "__main__":
    main()
