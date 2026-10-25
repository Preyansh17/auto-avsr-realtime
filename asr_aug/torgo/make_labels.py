"""TORGO real-data label CSVs in the repo format (dataset_name,rel_path,frames,token_ids) under one combined root.

    python make_labels.py   # -> data/{train,val,test}_spm1023.csv, root/torgo_raw -> TORGO raw (read-only symlink)

frames = round(duration * 25) as for the patient rows. Synthetic rows are added later under root/synth_higgs_tg.
"""
import csv
import os

import sentencepiece as spm

T = "/scratch/pa2753/torgo_aug"
RAW = "/scratch/th3482/TORGO/raw"
SPM = "/home/pa2753/auto-avsr-realtime/spm/spm_unigram_1023.model"


def main() -> None:
    os.makedirs(f"{T}/root", exist_ok=True)
    for name, target in (("torgo_raw", RAW), ("synth_higgs_tg", "/scratch/pa2753/gen_runs")):
        if not os.path.islink(f"{T}/root/{name}"):
            os.symlink(target, f"{T}/root/{name}")
    sp = spm.SentencePieceProcessor(model_file=SPM)
    rows = {"train": [], "val": [], "test": []}
    for i in csv.DictReader(open(f"{T}/data/items.csv")):
        rel = os.path.relpath(i["wav"], RAW)
        assert "," not in rel
        ids = sp.encode(i["text"])
        assert sp.decode(ids) == i["text"]
        rows[i["split"]].append(f"torgo_raw,{rel},{max(1, round(float(i['duration']) * 25))},{' '.join(map(str, ids))}")
    for s, r in rows.items():
        open(f"{T}/data/{s}_spm1023.csv", "w").write("\n".join(r) + "\n")
        print(s, len(r))


if __name__ == "__main__":
    main()
