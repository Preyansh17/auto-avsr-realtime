"""Pre-registered tests for the re-voicing study (README.md) and the Nemotron LR probe.

    python report_rv.py --model whisper --arms V1 V2 --seeds 4-10 [--ckpt best|last] [--out f.md]
    python report_rv.py --model nemotron --runs-dir /scratch/pa2753/nemo_lr/runs --arms R_flat1e-4 ... --ref R_noam

Per arm vs the reference arm: per-seed corpus test WER, mean +- sd, delta, two-sided Welch t-test over seeds, and a
sentence-level paired bootstrap 95% interval (errors averaged over seeds per sentence, 10000 draws). Also the
known-word / unseen-word subsets (asr_aug audit.json forbidden words), no claim. Seeds are matched by number only
through --seeds; arms may have different seed sets (Welch is unpaired).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/home/pa2753/auto-avsr-realtime")
sys.path.insert(0, "/scratch/pa2753/asr_aug/code")
from report import load  # noqa: E402  (patient report: same alignment + normalisation)

BASE = "/scratch/pa2753/asr_aug"


def seeds_arg(s):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--runs-dir", default=f"{BASE}/runs")
    ap.add_argument("--ref", default="R")
    ap.add_argument("--ref-runs-dir", default=None, help="where the reference arm lives (default --runs-dir)")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--ref-seeds", default=None)
    ap.add_argument("--ckpt", choices=["best", "last"], default="best")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sfx = "" if a.ckpt == "best" else "_last"
    forbidden = set(json.load(open(f"{BASE}/data/audit.json"))["forbidden_words"])

    def runs_of(arm, d, keep):
        out = {}
        for p in sorted(glob.glob(f"{d}/{a.model}_{arm}_seed*")):
            m = re.match(rf"{a.model}_{re.escape(arm)}_seed(\d+)$", os.path.basename(p))
            f = f"{p}/test_hyps{sfx}.tsv"
            if m and os.path.exists(f) and (keep is None or int(m.group(1)) in keep):
                out[int(m.group(1))] = load(f)
        return out

    ref = runs_of(a.ref, a.ref_runs_dir or a.runs_dir, seeds_arg(a.ref_seeds or a.seeds))
    assert ref, f"no {a.ref} runs"
    utts = sorted(next(iter(ref.values())))
    first = next(iter(ref.values()))
    unseen = [u for u in utts if forbidden & set(first[u]["ref"])]
    subsets = {"all": utts, "known": [u for u in utts if u not in unseen], "unseen": unseen}
    wer = lambda run, us: 100 * sum(run[u]["err"] for u in us) / sum(run[u]["n"] for u in us)  # noqa: E731
    rng = np.random.default_rng(0)
    L = [f"# {a.model} test WER % ({a.ckpt} checkpoint) vs {a.ref} (seeds {sorted(ref)})", "",
         "| arm | seeds | subset | arm WER mean +- sd | ref WER mean +- sd | delta | Welch p | bootstrap 95% | per-seed |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm in a.arms:
        runs = runs_of(arm, a.runs_dir, seeds_arg(a.seeds))
        if not runs:
            L.append(f"| {arm} | 0 | - | no runs | | | | | |")
            continue
        for run in runs.values():
            assert sorted(run) == utts, f"{arm}: test clip set differs from {a.ref}"
            assert all(run[u]["ref"] == first[u]["ref"] for u in utts), f"{arm}: references differ"
        for k, us in subsets.items():
            x = [wer(r, us) for r in runs.values()]
            y = [wer(r, us) for r in ref.values()]
            p = stats.ttest_ind(x, y, equal_var=False).pvalue if len(x) > 1 and len(y) > 1 else float("nan")
            e = np.array([np.mean([r[u]["err"] for r in runs.values()]) for u in us])
            e0 = np.array([np.mean([r[u]["err"] for r in ref.values()]) for u in us])
            n = np.array([first[u]["n"] for u in us], dtype=float)
            idx = rng.integers(0, len(us), size=(a.boot, len(us)))
            dist = 100 * (e[idx].sum(1) - e0[idx].sum(1)) / n[idx].sum(1)
            sd = lambda v: f" +- {statistics.stdev(v):.2f}" if len(v) > 1 else ""  # noqa: E731
            L.append(f"| {arm} | {sorted(runs)} | {k} ({len(us)}) | {statistics.mean(x):.2f}{sd(x)} | "
                     f"{statistics.mean(y):.2f}{sd(y)} | {statistics.mean(x) - statistics.mean(y):+.2f} | {p:.4f} | "
                     f"[{np.percentile(dist, 2.5):+.2f}, {np.percentile(dist, 97.5):+.2f}] | "
                     f"{' '.join(f'{v:.2f}' for v in x) if k == 'all' else ''} |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
