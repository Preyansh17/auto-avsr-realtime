"""Leakage guard for the generator benchmark (asr_aug/gen/README.md, A0 check 4; PLAN item 5 hard rule).

Fails (exit 1) if any sentence to be generated equals, after normalisation, a sentence of any non-train role:
every val / test / oov / heldout / rainbow file in LATEST_combined/roles (s1, s2, s2ov, s2ovu, s2conv, s3*, s3ov, old_*).

    python check_leak.py TEXT [TEXT ...]   # TEXT = a .json list of {"norm"| "text"} rows, or a .csv/.tsv with a text column
"""
import csv
import glob
import json
import os
import re
import sys

ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"


def norm(t):  # same normaliser as asr_aug/stack/make_gen_plan.py
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def held_out():
    held = {}
    for f in sorted(glob.glob(f"{ROLES}/*.jsonl")):
        role = os.path.basename(f)[:-6]
        if role.endswith("_train"):
            continue
        for line in open(f):
            held.setdefault(norm(json.loads(line)["transcript"]), set()).add(role)
    return held


def sentences(path):
    if path.endswith(".json"):
        return [r["norm"] if "norm" in r else r["text"] for r in json.load(open(path))]
    rows = csv.DictReader(open(path), delimiter="\t" if path.endswith(".tsv") else ",")
    col = next(c for c in ("text", "transcript", "norm", "sentence") if c in rows.fieldnames)
    return [r[col] for r in rows]


def main():
    held = held_out()
    print(f"held-out sentences: {len(held)} from {len({r for v in held.values() for r in v})} roles")
    bad = 0
    for path in sys.argv[1:]:
        s = sentences(path)
        hits = [(t, sorted(held[norm(t)])) for t in s if norm(t) in held]
        bad += len(hits)
        print(f"{path}: {len(s)} sentences, {len(hits)} match a held-out sentence")
        for t, roles in hits[:20]:
            print(f"  LEAK {t!r} in {roles}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
