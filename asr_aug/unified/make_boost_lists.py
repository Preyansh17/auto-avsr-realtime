"""Phrase lists for NeMo GPU phrase boosting on Tianyu's semg2 sets (PLAN.md 3.3 item 0, Step 5 arm PB).

    python make_boost_lists.py --train /scratch/pa2753/unified/ft/manifests_v2r_check/train.json --out <dir>

clean.txt (the claim): the unique sentences of our real train set, text the model already trains on.
words.txt (ceiling): clean + every test-set word absent from the train vocabulary. Test words are known here, so it is
never the claim.
"""
import argparse
import csv
import json
import re

PKG = "/scratch/pa2753/tianyu_semg2_best_v1/semg2_best_v1_20261029"
TESTS = ["eval_s2_test", "eval_s2_oov", "eval_s2ov_test", "eval_conv34"]


def words(t): return re.findall(r"[A-Za-z0-9']+", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pkg", default=PKG)
    a = ap.parse_args()
    sents = sorted({" ".join(words(json.loads(l)["text"])) for l in open(a.train)} - {""})
    vocab = {w.lower() for s in sents for w in words(s)}
    new = {}
    for c in TESTS:
        for r in csv.DictReader(open(f"{a.pkg}/phone_streaming_nemotron/data/{c}.csv")):
            for w in words(r["text"]):
                if w.lower() not in vocab:
                    new.setdefault(w.lower(), w)  # first spelling seen
    open(f"{a.out}/clean.txt", "w").write("\n".join(sents) + "\n")
    open(f"{a.out}/words.txt", "w").write("\n".join(sents + sorted(new.values())) + "\n")
    print(f"clean: {len(sents)} sentences; words: + {len(new)} new test words")


if __name__ == "__main__":
    main()
