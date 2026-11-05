"""Tianyu's three Higgs pools as --extra-train-manifest files for his own semg.run code (PLAN.md 3.3 item 0, Step 6).

    python make_n3_extras.py --out /scratch/pa2753/n3/manifests

His recipe passes s2_higgsft_new150t_2k__ovfilt, s2_higgszs_med60_nocap_2k__ovfilt and s2_higgsft_trainrend8__ovfilt,
which we cannot read. The package train.csv lists the same 5,378 synthetic rows (plus his 189 real ones, which his
code reads from the split itself), so we rebuild the three manifests from it, in his order. Rows carry the fields his
_concat / prepare_manifest_label read: an absolute rel_path (the wav), dataset_name, frame_count, transcript.
"""
import argparse
import csv
import json
import os

import soundfile as sf

PKG = "/scratch/pa2753/tianyu_semg2_best_v1/semg2_best_v1_20261029"
POOLS = ["synth_wide_higgsft_new150t_s2", "synth_wide_higgszs_med60_nocap_s2", "synth_trainrend_higgsft_raw_s2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(f"{a.pkg}/phone_streaming_nemotron/data/train.csv")))
    for pool in POOLS:
        with open(f"{a.out}/{pool}.jsonl", "w") as f:
            n = 0
            for r in rows:
                if r["dataset"] != pool:
                    continue
                dur = sf.info(r["torch_path"]).duration
                f.write(json.dumps({"dataset_name": pool, "rel_path": r["torch_path"], "audio_path": r["torch_path"],
                                    "frame_count": round(dur * 25), "transcript": r["text"]}, sort_keys=True) + "\n")
                n += 1
        print(pool, n)
    real = sum(r["dataset"] not in POOLS for r in rows)
    print(f"real rows in train.csv (read from the split by his code, not written): {real}")


if __name__ == "__main__":
    main()
