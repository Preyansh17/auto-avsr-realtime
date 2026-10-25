"""Size of the Whisper decoder-prefix mismatch (review finding inference-1, 2026-10-20).

    python prefix_report.py --study patient --arms R M0.5 A0.5 D0.5 --seeds 4-10 --split test
    python prefix_report.py --study torgo --arms R M0.5 M0.5sc --seeds 1-7

For each best/ checkpoint: WER of the published hyps (<sot><en><transcribe><notimestamps>, <split>_hyps.tsv) and of
the re-decode with the trained prompt (<sot><sot><notimestamps>, <split>_hyps_sotsot.tsv, prefix_redecode.sbatch).
Per arm: mean +- sd under each prompt, the per-run change, and each arm's delta vs R under both prompts with the
pre-registered tests (two-sided Welch over seeds, sentence-level bootstrap 95% interval with errors averaged over
seeds, 10000 draws). Also the fraction of hyps carrying case or punctuation (the symptom of the mismatch). Note
best/ itself was chosen by val WER under the mismatched prompt; this measures the decode-time effect only.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/home/pa2753/auto-avsr-realtime")
sys.path.insert(0, "/scratch/pa2753/asr_aug/code")
from report import align, normalize  # noqa: E402  (patient report: same alignment + normalisation)

RUNS = {"patient": "/scratch/pa2753/asr_aug/runs", "torgo": "/scratch/pa2753/torgo_aug/runs"}
STYLED = re.compile(r"[A-Z.,!?;:]")


def load(tsv: str) -> tuple[dict[str, dict], float]:
    out, styled = {}, 0
    for line in open(tsv).read().splitlines()[1:]:
        path, ref, hyp = (line.split("\t") + ["", ""])[:3]
        r, h = normalize(ref).split(), normalize(hyp).split()
        s, d, i = align(r, h)
        out["/".join(path.split("/")[-4:])] = {"err": s + d + i, "n": len(r)}
        styled += bool(STYLED.search(hyp))
    return out, styled / max(1, len(out))


def seeds_arg(s: str) -> set[int]:
    out = set()
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out |= set(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=list(RUNS), required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ref", default="R")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keep = seeds_arg(a.seeds)
    prompts = {"published": f"{a.split}_hyps.tsv", "sotsot": f"{a.split}_hyps_sotsot.tsv"}

    def runs_of(arm):
        out = {}
        for p in sorted(glob.glob(f"{RUNS[a.study]}/whisper_{arm}_seed*")):
            m = re.match(rf"whisper_{re.escape(arm)}_seed(\d+)$", os.path.basename(p))
            if m and int(m.group(1)) in keep and all(os.path.exists(f"{p}/{f}") for f in prompts.values()):
                out[int(m.group(1))] = {k: load(f"{p}/{f}") for k, f in prompts.items()}
        return out

    data = {arm: runs_of(arm) for arm in dict.fromkeys([a.ref] + a.arms)}
    ref_runs = data[a.ref]
    assert ref_runs, f"no {a.ref} runs with both hyps files"
    utts = sorted(next(iter(ref_runs.values()))["published"][0])
    n = np.array([next(iter(ref_runs.values()))["published"][0][u]["n"] for u in utts], dtype=float)
    wer = lambda run: 100 * sum(v["err"] for v in run.values()) / sum(v["n"] for v in run.values())  # noqa: E731
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else float("nan")  # noqa: E731
    rng = np.random.default_rng(0)

    L = [f"# Whisper prefix re-decode, {a.study} {a.split}, best/ (seeds {sorted(keep)})", "",
         "| arm | runs | published WER | trained-prompt WER | change mean (min, max) | styled hyps pub -> sotsot |",
         "|---|---|---|---|---|---|"]
    for arm, runs in data.items():
        if not runs:
            L.append(f"| {arm} | 0 | | | | |")
            continue
        for r in runs.values():
            assert sorted(r["published"][0]) == utts and sorted(r["sotsot"][0]) == utts, f"{arm}: clip set differs"
        pub = [wer(r["published"][0]) for r in runs.values()]
        new = [wer(r["sotsot"][0]) for r in runs.values()]
        ch = [y - x for x, y in zip(pub, new)]
        st = [np.mean([r[k][1] for r in runs.values()]) for k in prompts]
        L.append(f"| {arm} | {len(runs)} | {statistics.mean(pub):.2f} +- {sd(pub):.2f} | {statistics.mean(new):.2f} +- "
                 f"{sd(new):.2f} | {statistics.mean(ch):+.2f} ({min(ch):+.2f}, {max(ch):+.2f}) | "
                 f"{st[0]:.2f} -> {st[1]:.2f} |")
    L += ["", f"Deltas vs {a.ref}:", "",
          "| arm | prompt | delta | Welch p | sentence bootstrap 95% |", "|---|---|---|---|---|"]
    for arm in a.arms:
        if arm == a.ref or not data[arm]:
            continue
        for k in prompts:
            x = [wer(r[k][0]) for r in data[arm].values()]
            y = [wer(r[k][0]) for r in ref_runs.values()]
            p = stats.ttest_ind(x, y, equal_var=False).pvalue
            e = np.array([np.mean([r[k][0][u]["err"] for r in data[arm].values()]) for u in utts])
            e0 = np.array([np.mean([r[k][0][u]["err"] for r in ref_runs.values()]) for u in utts])
            idx = rng.integers(0, len(utts), size=(a.boot, len(utts)))
            dist = 100 * (e[idx].sum(1) - e0[idx].sum(1)) / n[idx].sum(1)
            L.append(f"| {arm} | {k} | {statistics.mean(x) - statistics.mean(y):+.2f} | {p:.4f} | "
                     f"[{np.percentile(dist, 2.5):+.2f}, {np.percentile(dist, 97.5):+.2f}] |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
