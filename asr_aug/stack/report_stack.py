"""Item-3 stack report (pre-registered in asr_aug/stack/README.md). Writes MBR hyps, prints the tables and contrasts.

    python report_stack.py [--seeds 1 2 3 4 5]

Systems per arm: single seeds (mean +- sd), soup (one model, runs/<arm>_soup), MBR vote over the seeds' hyps
(per clip, the hypothesis with the lowest summed word edit distance to the other seeds' hypotheses; ties -> lowest seed).
Refs come from the shared role files (as in shared_eval/score_shared.py). Contrasts are paired two-level bootstraps:
seeds resampled within each system that has several, sentences (unique ref texts) resampled jointly; 4000 draws.
"""
import argparse
import csv
import json
import os
import random
import re
import statistics as st

R = "/scratch/pa2753/stack/runs"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
SETS = [f"{p}_{s}" for p in ("s1", "s2") for s in ("val", "test", "oov")]


def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def refs(role):
    return {x["take_id"]: norm(x["transcript"]) for x in map(json.loads, open(f"{ROLES}/{role}.jsonl"))}


def hyps(tsv):
    return {os.path.splitext(os.path.basename(r["path"]))[0]: norm(r["hyp"] or "")
            for r in csv.DictReader(open(tsv), delimiter="\t")}


def mbr(hs):
    out = {}
    for k in hs[0]:
        cand = [h[k] for h in hs]
        cost = [sum(ed(c.split(), o.split()) for o in cand) for c in cand]
        out[k] = cand[cost.index(min(cost))]
    return out


def errs(ref, h):  # per take: (sentence, errors, words)
    return {k: (r, ed(r.split(), h[k].split()), len(r.split())) for k, r in ref.items()}


def wer(e): return 100 * sum(x[1] for x in e.values()) / sum(x[2] for x in e.values())


def boot(A, B, rng, n=4000):
    """A, B: lists (one per seed) of per-take error dicts on the same takes. Returns diff, CI, p."""
    by_sent = {}
    for k, (s, _, _) in A[0].items():
        by_sent.setdefault(s, []).append(k)
    sents = sorted(by_sent)
    d0 = st.mean(map(wer, A)) - st.mean(map(wer, B))
    bs = []
    for _ in range(n):
        pick = [k for s in (rng.choice(sents) for _ in sents) for k in by_sent[s]]
        f = lambda S: st.mean(100 * sum(e[k][1] for k in pick) / sum(e[k][2] for k in pick)
                              for e in (rng.choice(S) for _ in S))
        bs.append(f(A) - f(B))
    bs.sort()
    p = 2 * min(sum(b >= 0 for b in bs), sum(b <= 0 for b in bs)) / n
    return d0, bs[int(0.025 * n)], bs[int(0.975 * n)], min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    rng = random.Random(0)
    for kind in ("", "_last"):
        print(f"\n== {'best (val-selected)' if not kind else 'last epoch'} checkpoint")
        E = {}
        for arm in ("E1", "E2"):
            for s in SETS:
                ref = refs(s)
                hs = [hyps(f"{R}/{arm}_seed{x}/{s}_hyps{kind}.tsv") for x in a.seeds]
                E[arm, "single", s] = [errs(ref, h) for h in hs]
                m = mbr(hs)
                with open(f"{R}/{arm}_mbr_{s}_hyps{kind}.tsv", "w") as f:
                    f.write("take_id\thyp\n" + "".join(f"{k}\t{v}\n" for k, v in m.items()))
                E[arm, "mbr", s] = [errs(ref, m)]
                sp = f"{R}/{arm}_soup/{s}_hyps.tsv"
                if not kind and os.path.exists(sp):
                    E[arm, "soup", s] = [errs(ref, hyps(sp))]
        print("| arm | system | " + " | ".join(SETS) + " |\n|---|---|" + "---|" * len(SETS))
        for arm in ("E1", "E2"):
            for sysn in ("single", "soup", "mbr"):
                if (arm, sysn, SETS[0]) not in E:
                    continue
                cells = []
                for s in SETS:
                    v = [wer(e) for e in E[arm, sysn, s]]
                    cells.append(f"{st.mean(v):.2f} ± {st.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}")
                print(f"| {arm} | {sysn} | " + " | ".join(cells) + " |")
        print("\nPRIMARY E2 - E1, single seeds, s1_oov:" if not kind else "\nE2 - E1, single seeds, s1_oov (last):")
        print("  diff %+.2f pp 95%% CI [%+.2f, %+.2f] p=%.4f" % boot(E["E2", "single", "s1_oov"], E["E1", "single", "s1_oov"], rng))
        print("secondary E2 - E1 (single seeds):")
        for s in ("s1_test", "s2_test", "s2_oov"):
            print(f"  {s}: diff %+.2f [%+.2f, %+.2f] p=%.4f" % boot(E["E2", "single", s], E["E1", "single", s], rng))
        print("secondary ensemble vs single-seed mean (s1_oov, s1_test):")
        for arm in ("E1", "E2"):
            for sysn in ("soup", "mbr"):
                if (arm, sysn, "s1_oov") in E:
                    for s in ("s1_oov", "s1_test"):
                        print(f"  {arm} {sysn} - single {s}: diff %+.2f [%+.2f, %+.2f] p=%.4f"
                              % boot(E[arm, sysn, s], E[arm, "single", s], rng))


if __name__ == "__main__":
    main()
