"""Augmentation results table: test WER per arm on the full split_v1 test set and on its known-word / unseen-word subsets.

    python report.py [--model whisper] [--ref-arm R] [--out report.md]

Reads runs/<model>_<arm>_seed<s>/{val,test}_hyps.tsv. Subsets: a test sentence is "unseen-word" if it contains a
word that never occurs in ASR train (audit.json forbidden words, which no synthetic sentence contains either), else
"known-word". Per arm: WER mean +- sd over seeds; paired bootstrap (over sentences, errors averaged over seeds)
of WER(arm) - WER(ref arm) with a 95% interval; del/ins/sub; for the unseen subset, how often the unseen word itself
is in the hypothesis.

--seeds 4-10 restricts every arm to those seeds. --ckpt last reads the final-epoch decodes instead of best/. --boot2 adds the pre-registered confirmatory block (README, 2026-10-13):
for each arm vs the reference, WER difference of seed-averaged WERs with a two-level bootstrap (seeds resampled within each
arm independently, sentences resampled jointly), a two-sided bootstrap p-value and a Holm decision over the compared arms.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

BASE = "/scratch/pa2753/asr_aug"
sys.path.insert(0, "/home/pa2753/auto-avsr-realtime")
from asr_baselines.metrics import normalize  # noqa: E402


def align(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of a minimum-edit word alignment."""
    n, m = len(ref), len(hyp)
    d = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(m + 1):
        d[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, s, de, ins = d[i - 1][j - 1]
            sub = (c + (ref[i - 1] != hyp[j - 1]), s + (ref[i - 1] != hyp[j - 1]), de, ins)
            c, s, de, ins = d[i - 1][j]
            dele = (c + 1, s, de + 1, ins)
            c, s, de, ins = d[i][j - 1]
            inse = (c + 1, s, de, ins + 1)
            d[i][j] = min(sub, dele, inse)
    return d[n][m][1:]


def load(tsv: str) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(open(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        ref, hyp = normalize(r["ref"]).split(), normalize(r["hyp"]).split()
        s, de, i = align(ref, hyp)
        out[os.path.basename(r["path"])] = {"ref": ref, "hyp": hyp, "err": s + de + i, "n": len(ref), "sub": s,
                                            "del": de, "ins": i}
    return out


def wer(rows) -> float:
    return sum(r["err"] for r in rows) / max(1, sum(r["n"] for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--ref-arm", default="R")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seeds", default=None, help="e.g. 4-10 or 1,2,3")
    ap.add_argument("--boot2", nargs="*", default=None, help="arms for the two-level bootstrap + Holm block")
    ap.add_argument("--ckpt", choices=["best", "last"], default="best",
                    help="last = {val,test}_hyps_last.tsv, the final-epoch checkpoint (val-free secondary, seeds 4-10)")
    a = ap.parse_args()
    keep = None
    if a.seeds:
        keep = set()
        for part in a.seeds.split(","):
            lo, _, hi = part.partition("-")
            keep |= set(range(int(lo), int(hi or lo) + 1))
    forbidden = set(json.load(open(f"{BASE}/data/audit.json"))["forbidden_words"])
    sfx = "" if a.ckpt == "best" else "_last"
    runs = collections.defaultdict(dict)
    for d in sorted(glob.glob(f"{BASE}/runs/{a.model}_*_seed*")):
        m = re.match(rf"{a.model}_(.+)_seed(\d+)$", os.path.basename(d))
        if m and os.path.exists(f"{d}/test_hyps{sfx}.tsv") and (keep is None or int(m.group(2)) in keep):
            runs[m.group(1)][int(m.group(2))] = {"test": load(f"{d}/test_hyps{sfx}.tsv"),
                                                 "val": load(f"{d}/val_hyps{sfx}.tsv") if os.path.exists(f"{d}/val_hyps{sfx}.tsv") else None}
    assert a.ref_arm in runs, f"no {a.ref_arm} runs"
    utts = sorted(next(iter(runs[a.ref_arm].values()))["test"])
    first = next(iter(runs[a.ref_arm].values()))["test"]
    unseen = [u for u in utts if forbidden & set(first[u]["ref"])]
    subsets = {"all": utts, "known": [u for u in utts if u not in unseen], "unseen": unseen}
    rng = np.random.default_rng(0)

    def per_sentence(arm, us):  # seed-averaged errors and word counts per sentence
        seeds = runs[arm].values()
        return (np.array([np.mean([s["test"][u]["err"] for s in seeds]) for u in us]),
                np.array([first[u]["n"] for u in us], dtype=float))

    L = [f"# {a.model}: synthetic augmentation results ({a.ckpt} checkpoint, seeds {a.seeds or 'all available'})", "",
         f"Test = split_v1 merged ({len(utts)} clips): known-word {len(subsets['known'])}, unseen-word {len(unseen)} "
         f"(contain {sorted(forbidden & {w for u in unseen for w in first[u]['ref']})}). WER in %, mean +- sd over seeds; "
         f"delta = paired bootstrap vs {a.ref_arm} over sentences (seed-averaged), 95% interval. The unseen subset has "
         f"{len(unseen)} sentences: indicative only.", "",
         "| arm | seeds | val | test all | delta all | known | delta known | unseen | delta unseen | unseen-word hit | S/D/I (all, mean) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    ref_ps = {k: per_sentence(a.ref_arm, us) for k, us in subsets.items()}

    def order(arm):
        return (arm != a.ref_arm, arm != "S", arm)

    for arm in sorted(runs, key=order):
        seeds = runs[arm]
        cells = []
        vals = [100 * wer(s["val"].values()) for s in seeds.values() if s["val"]]
        cells.append(f"{statistics.mean(vals):.2f}" if vals else "-")
        for k, us in subsets.items():
            w = [100 * wer([s["test"][u] for u in us]) for s in seeds.values()]
            cells.append(f"{statistics.mean(w):.2f} +- {statistics.stdev(w) if len(w) > 1 else 0:.2f}")
            if arm == a.ref_arm:
                cells.append("ref")
                continue
            e, n = per_sentence(arm, us)
            e0, _ = ref_ps[k]
            idx = rng.integers(0, len(us), size=(a.boot, len(us)))
            dist = 100 * ((e[idx].sum(1) - e0[idx].sum(1)) / n[idx].sum(1))
            point = 100 * (e.sum() - e0.sum()) / n.sum()
            cells.append(f"{point:+.2f} [{np.percentile(dist, 2.5):+.2f}, {np.percentile(dist, 97.5):+.2f}]")
        hits = [any(w in s["test"][u]["hyp"] for w in forbidden & set(first[u]["ref"])) for s in seeds.values() for u in unseen]
        cells.append(f"{sum(hits)}/{len(hits)}")
        sdi = [statistics.mean(sum(s["test"][u][x] for u in utts) for s in seeds.values()) for x in ("sub", "del", "ins")]
        cells.append("/".join(f"{v:.1f}" for v in sdi))
        L.append(f"| {arm} | {len(seeds)} | " + " | ".join(cells) + " |")
    if a.boot2:
        L += ["", f"## Two-level bootstrap vs {a.ref_arm} (seeds {a.seeds or 'all'}; {a.boot} draws; Holm over {len(a.boot2)} arms)", "",
              "| arm | seeds | ref seeds | delta all (pp) | interval at Holm level (1 - t) | p (two-sided) | Holm threshold t | helps? |",
              "|---|---|---|---|---|---|---|---|"]
        n_w = np.array([first[u]["n"] for u in utts], dtype=float)

        def err_matrix(arm):  # seeds x sentences
            return np.array([[s["test"][u]["err"] for u in utts] for s in runs[arm].values()], dtype=float)

        E0 = err_matrix(a.ref_arm)
        rng2 = np.random.default_rng(1)
        res = []
        for arm in a.boot2:
            E = err_matrix(arm)
            point = 100 * (E.mean(0).sum() - E0.mean(0).sum()) / n_w.sum()
            dist = np.empty(a.boot)
            for b in range(a.boot):
                si = rng2.integers(0, len(E), len(E))
                s0 = rng2.integers(0, len(E0), len(E0))
                ui = rng2.integers(0, len(utts), len(utts))
                dist[b] = 100 * (E[si][:, ui].mean(0).sum() - E0[s0][:, ui].mean(0).sum()) / n_w[ui].sum()
            p = min(1.0, 2 * min((dist >= 0).mean(), (dist <= 0).mean()))
            res.append((arm, len(E), point, dist, p))
        order_p = sorted(range(len(res)), key=lambda i: res[i][4])
        thr, stop = {}, False
        for k, i in enumerate(order_p):  # Holm step-down
            t = 0.05 / (len(res) - k)
            thr[i] = (t, (not stop) and res[i][4] < t and res[i][2] < 0)
            if not (res[i][4] < t):
                stop = True
        for i, (arm, ns, point, dist, p) in enumerate(res):
            t, ok = thr[i]
            lo, hi = np.percentile(dist, [100 * t / 2, 100 * (1 - t / 2)])
            L.append(f"| {arm} | {ns} | {len(E0)} | {point:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.4f} | {t:.4f} | "
                     f"{'yes' if ok else 'no detectable effect'} |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
