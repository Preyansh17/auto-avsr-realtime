"""TORGO augmentation results (pre-registered tests in README.md).

    python report.py [--model whisper] [--ckpt best|last] [--seeds 1-7] [--out report.md]

Per arm: test WER on all clips, sentences only, single words only, per severity and per speaker (mean +- sd over seeds).
For each arm vs R: seed-level Welch t-test on per-seed corpus WER, and a sentence-level paired bootstrap 95% interval
(errors averaged over seeds per clip, 10000 draws). Zero-shot whisper-large-v3 (data/zeroshot_test.tsv) as a reference row.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import statistics
import sys

import numpy as np
from scipy import stats

T = "/scratch/pa2753/torgo_aug"
sys.path.insert(0, "/home/pa2753/auto-avsr-realtime")
sys.path.insert(0, "/scratch/pa2753/asr_aug/code")
from report import align, normalize  # noqa: E402  (patient report: same alignment + normalisation)

SEV = {"F01": "severe", "M01": "severe", "M02": "severe", "M04": "severe",
       "F03": "mild", "F04": "mild", "M03": "mild", "M05": "mild"}


def load(tsv: str) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        ref, hyp = normalize(r["ref"]).split(), normalize(r["hyp"]).split()
        s, d, i = align(ref, hyp)
        key = "/".join(r["path"].split("/")[-4:])  # spk/Session/wav_headMic/xxxx.wav
        out[key] = {"err": s + d + i, "n": len(ref)}
    return out


def wer(rows) -> float:
    return 100 * sum(r["err"] for r in rows) / max(1, sum(r["n"] for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--ckpt", choices=["best", "last"], default="best")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keep = None
    if a.seeds:
        lo, _, hi = a.seeds.partition("-")
        keep = set(range(int(lo), int(hi or lo) + 1))
    sfx = "" if a.ckpt == "best" else "_last"
    runs = collections.defaultdict(dict)
    for d in sorted(glob.glob(f"{T}/runs/{a.model}_*_seed*")):
        m = re.match(rf"{a.model}_(.+)_seed(\d+)$", os.path.basename(d))
        if m and os.path.exists(f"{d}/test_hyps{sfx}.tsv") and (keep is None or int(m.group(2)) in keep):
            runs[m.group(1)][int(m.group(2))] = load(f"{d}/test_hyps{sfx}.tsv")
    meta = {"/".join(os.path.relpath(i["wav"], "/scratch/th3482/TORGO/raw").split("/")): i
            for i in csv.DictReader(open(f"{T}/data/items.csv")) if i["split"] == "test"}
    utts = sorted(meta)
    subsets = {"all": utts, "sentence": [u for u in utts if meta[u]["type"] == "sentence"],
               "word": [u for u in utts if meta[u]["type"] == "word"],
               "mild": [u for u in utts if SEV[meta[u]["speaker"]] == "mild"],
               "severe": [u for u in utts if SEV[meta[u]["speaker"]] == "severe"]}
    for spk in sorted(SEV):
        subsets[spk] = [u for u in utts if meta[u]["speaker"] == spk]
    zs = f"{T}/data/zeroshot_test.tsv"
    if os.path.exists(zs) and a.model == "whisper":
        runs["zero-shot"] = {0: load(zs)}
    rng = np.random.default_rng(0)
    L = [f"# TORGO {a.model}: test WER % ({a.ckpt} checkpoint, seeds {a.seeds or 'all'})", "",
         "n clips: " + ", ".join(f"{k} {len(v)}" for k, v in subsets.items()), "",
         "| arm | seeds | " + " | ".join(subsets) + " |", "|---|---|" + "---|" * len(subsets)]
    for arm in sorted(runs, key=lambda x: (x != "R", x == "zero-shot", x)):
        cells = []
        for k, us in subsets.items():
            w = [wer([s[u] for u in us]) for s in runs[arm].values()]
            cells.append(f"{statistics.mean(w):.2f}" + (f" +- {statistics.stdev(w):.2f}" if len(w) > 1 else ""))
        L.append(f"| {arm} | {len(runs[arm])} | " + " | ".join(cells) + " |")
    if "R" in runs:
        L += ["", "## vs R", "", "| arm | subset | delta (pp) | Welch p (seeds) | bootstrap 95% (sentences) |", "|---|---|---|---|---|"]
        for arm in sorted(runs):
            if arm in ("R", "zero-shot"):
                continue
            for k in ("all", "sentence", "word", "mild", "severe"):
                us = subsets[k]
                x = [wer([s[u] for u in us]) for s in runs[arm].values()]
                r = [wer([s[u] for u in us]) for s in runs["R"].values()]
                p = stats.ttest_ind(x, r, equal_var=False).pvalue if len(x) > 1 and len(r) > 1 else float("nan")
                e = np.array([np.mean([s[u]["err"] for s in runs[arm].values()]) for u in us])
                e0 = np.array([np.mean([s[u]["err"] for s in runs["R"].values()]) for u in us])
                n = np.array([meta and next(iter(runs["R"].values()))[u]["n"] for u in us], dtype=float)
                idx = rng.integers(0, len(us), size=(a.boot, len(us)))
                dist = 100 * (e[idx].sum(1) - e0[idx].sum(1)) / n[idx].sum(1)
                L.append(f"| {arm} | {k} | {statistics.mean(x) - statistics.mean(r):+.2f} | {p:.4f} | "
                         f"[{np.percentile(dist, 2.5):+.2f}, {np.percentile(dist, 97.5):+.2f}] |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
