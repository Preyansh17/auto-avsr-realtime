"""Item 8 report: SGEM TTA vs no TTA (step 0), pre-registered in asr_aug/tta/README.md.

    python report_tta.py [--seeds 1 2 3 4 5]

Selection per arm: (scope, step) in {ln, feat} x {1, 3, 5, 10} with the lowest pooled s1_val + s2_val WER (mean over
seeds); ties -> fewer steps, then ln. Primary (Holm over 2), TTA - step 0, paired two-level bootstrap:
  1. E2 on s1_oov.   2. E1 on semg2, s2_test + s2_oov pooled.
"TTA helps" = 95% CI upper bound < 0 and Holm p < 0.05. Everything else is descriptive.
"""
import argparse
import os
import random
import statistics as st
import sys

sys.path.insert(0, "/scratch/pa2753/stack/code")
from report_stack import R, SETS, boot, errs, hyps, refs, wer  # noqa: E402

SCOPES, STEPS = ("ln", "feat"), (1, 3, 5, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--tag", default="tta")
    a = ap.parse_args()
    rng = random.Random(0)
    ref = {s: refs(s) for s in SETS}
    ref["s2_eval"] = {**ref["s2_test"], **ref["s2_oov"]}

    def E(arm, sc, k, s, seeds=a.seeds):
        if s == "s2_eval":
            return [{**e1, **e2} for e1, e2 in zip(E(arm, sc, k, "s2_test", seeds), E(arm, sc, k, "s2_oov", seeds))]
        return [errs(ref[s], hyps(f"{R}/{arm}_seed{x}/{a.tag}/{sc}/step{k}_{s}_hyps.tsv")) for x in seeds]

    def loops(arm, sc, k, s):
        n = 0
        for x in a.seeds:
            for line in list(open(f"{R}/{arm}_seed{x}/{a.tag}/{sc}/step{k}_{s}_hyps.tsv"))[1:]:
                _, r, h = line.rstrip("\n").split("\t")
                n += len(h) > 3 * max(1, len(r))
        return n

    sel = {}
    for arm in ("E1", "E2"):
        val = {}
        for sc in SCOPES:
            for k in (0,) + STEPS:
                v = [wer({**e1, **e2}) for e1, e2 in zip(E(arm, sc, k, "s1_val"), E(arm, sc, k, "s2_val"))]
                val[sc, k] = st.mean(v)
        sel[arm] = min(((sc, k) for sc in SCOPES for k in STEPS), key=lambda c: (round(val[c], 6), c[1], SCOPES.index(c[0])))
        print(f"\n== {arm}: pooled val WER (mean over seeds), step 0 = {val['ln', 0]:.2f} (ln) / {val['feat', 0]:.2f} (feat)")
        print("| scope | " + " | ".join(f"step {k}" for k in STEPS) + " |\n|---|" + "---|" * len(STEPS))
        for sc in SCOPES:
            print(f"| {sc} | " + " | ".join(f"{val[sc, k]:.2f}" for k in STEPS) + " |")
        sc, k = sel[arm]
        print(f"selected: {sc} step {k} (val {val[sc, k]:.2f} vs step 0 {val[sc, 0]:.2f})")
        print("| set | step 0 | selected | oracle best step (descriptive) | loops step0 / selected |\n|---|---|---|---|---|")
        for s in SETS + ["s2_eval"]:
            w0 = st.mean(map(wer, E(arm, sc, 0, s)))
            ws = st.mean(map(wer, E(arm, sc, k, s)))
            orc = min(((c, q) for c in SCOPES for q in STEPS), key=lambda cq: st.mean(map(wer, E(arm, cq[0], cq[1], s))))
            wo = st.mean(map(wer, E(arm, orc[0], orc[1], s)))
            lp = "" if s == "s2_eval" else f"{loops(arm, sc, 0, s)} / {loops(arm, sc, k, s)}"
            print(f"| {s} | {w0:.2f} | {ws:.2f} | {wo:.2f} ({orc[0]} {orc[1]}) | {lp} |")

    print("\nPRIMARY (TTA - step 0, selected system, single seeds; Holm over 2)")
    prim = []
    for name, arm, s in (("1: E2 s1_oov", "E2", "s1_oov"), ("2: E1 s2_test+s2_oov (unseen patient)", "E1", "s2_eval")):
        sc, k = sel[arm]
        d, lo, hi, p = boot(E(arm, sc, k, s), E(arm, sc, 0, s), rng)
        prim.append((name, d, lo, hi, p))
    ps = sorted(range(2), key=lambda i: prim[i][4])
    holm = {ps[0]: min(1, 2 * prim[ps[0]][4])}
    holm[ps[1]] = min(1, max(holm[ps[0]], prim[ps[1]][4]))
    for i, (name, d, lo, hi, p) in enumerate(prim):
        verdict = "HELPS" if hi < 0 and holm[i] < 0.05 else "no claim"
        print(f"  {name}: {d:+.2f} pp [{lo:+.2f}, {hi:+.2f}] p={p:.4f} Holm={holm[i]:.4f} -> {verdict}")

    print("\nsecondary (TTA - step 0, selected system per arm)")
    for arm in ("E1", "E2"):
        sc, k = sel[arm]
        for s in SETS[1:3] + SETS[4:]:
            if (arm, s) in (("E2", "s1_oov"),):
                continue
            d, lo, hi, p = boot(E(arm, sc, k, s), E(arm, sc, 0, s), rng)
            print(f"  {arm} {s}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.4f}")


if __name__ == "__main__":
    main()
