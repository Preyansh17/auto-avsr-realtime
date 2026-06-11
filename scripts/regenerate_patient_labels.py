#!/usr/bin/env python3
"""Re-tokenize auto-avsr patient label CSVs for the streaming recipe.

The original CSVs ("dataset_name,rel_path,input_length,token ids") carry
token ids from the offline unigram5000 SentencePiece model. The streaming
Emformer RNN-T uses the 1023-piece model, so each row is decoded with the
old model and re-encoded with the new one. Output keeps the same 4-field
format with a _spm1023 filename suffix.

Usage:
  python scripts/regenerate_patient_labels.py --old-csv labels/train.csv --preview 5
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from online_avsr.text import extract_reference_from_filename  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-csv", required=True, nargs="+", help="Old label CSV(s) with unigram5000 token ids")
    parser.add_argument("--old-spm", default=os.path.join(PROJECT_ROOT, "spm", "unigram", "unigram5000.model"))
    parser.add_argument("--new-spm", default=os.path.join(PROJECT_ROOT, "spm", "spm_unigram_1023.model"))
    parser.add_argument("--out", default=None, help="Output path (single input only; default: <input>_spm1023.csv)")
    parser.add_argument("--text-from", choices=["tokens", "filename"], default="tokens",
                        help="Recover text from old token ids (default) or from the filename")
    parser.add_argument("--preview", type=int, default=0, help="Print the first N round-tripped rows and exit")
    return parser.parse_args()


def convert(old_csv, old_sp, new_sp, args):
    out_path = args.out or (os.path.splitext(old_csv)[0] + "_spm1023.csv")
    rows_out, skipped = [], 0
    previewed = 0
    with open(old_csv, encoding="utf-8") as f:
        for line_no, line in enumerate(f.read().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 4:
                skipped += 1
                continue
            dataset_name, rel_path, input_length = parts[0], parts[1], parts[2]
            if args.text_from == "filename":
                text = extract_reference_from_filename(rel_path).lower()
            else:
                old_ids = [int(t) for t in parts[3].split()]
                text = old_sp.decode(old_ids).lower()
            new_ids = new_sp.encode(text)
            if not text or not new_ids:
                skipped += 1
                continue
            if args.preview and previewed < args.preview:
                roundtrip = new_sp.decode(new_ids)
                print(f"[{line_no}] {rel_path}\n  text:      {text}\n  roundtrip: {roundtrip}")
                previewed += 1
            rows_out.append(f"{dataset_name},{rel_path},{input_length},{' '.join(map(str, new_ids))}")

    if args.preview:
        print(f"(preview only; {len(rows_out)} rows would be written, {skipped} skipped)")
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows_out) + "\n")
    print(f"{old_csv}: wrote {len(rows_out)} rows to {out_path} ({skipped} skipped)")


def main():
    args = parse_args()
    if args.out and len(args.old_csv) > 1:
        raise SystemExit("--out is only valid with a single --old-csv")
    import sentencepiece as spm

    old_sp = spm.SentencePieceProcessor(model_file=args.old_spm)
    new_sp = spm.SentencePieceProcessor(model_file=args.new_spm)
    if new_sp.get_piece_size() != 1023:
        raise SystemExit(f"--new-spm must be the 1023-piece model, got {new_sp.get_piece_size()}")
    for old_csv in args.old_csv:
        convert(old_csv, old_sp, new_sp, args)


if __name__ == "__main__":
    main()
