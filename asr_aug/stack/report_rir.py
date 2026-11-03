"""E2R report: room-simulated synthetic (E2R) vs plain synthetic (E2), pre-registered in README.md ("E2R").

    python report_rir.py [--seeds 1 2 3 4 5]

Primary: E2R - E2 on s1_oov, single seeds, val-selected checkpoint, paired two-level bootstrap (report_stack.boot).
"E2R helps" = 95% CI upper bound < 0. Secondary (CIs, no claims): s1_test, s2_test, s2_oov; last checkpoint;
E2R soup vs E2 soup (one model each, sentence bootstrap only).
"""
import argparse
import os
import random
import statistics as st

from report_stack import R, SETS, boot, errs, hyps, refs, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    rng = random.Random(0)
    for kind in ("", "_last"):
        print(f"\n== {'best (val-selected)' if not kind else 'last epoch'} checkpoint")
        E = {}
        for arm in ("E2", "E2R"):
            for s in SETS:
                ref = refs(s)
                E[arm, "single", s] = [errs(ref, hyps(f"{R}/{arm}_seed{x}/{s}_hyps{kind}.tsv")) for x in a.seeds]
                sp = f"{R}/{arm}_soup/{s}_hyps.tsv"
                if not kind and os.path.exists(sp):
                    E[arm, "soup", s] = [errs(ref, hyps(sp))]
        print("| arm | system | " + " | ".join(SETS) + " |\n|---|---|" + "---|" * len(SETS))
        for arm in ("E2", "E2R"):
            for sysn in ("single", "soup"):
                if (arm, sysn, SETS[0]) in E:
                    v = [[wer(e) for e in E[arm, sysn, s]] for s in SETS]
                    print(f"| {arm} | {sysn} | " + " | ".join(
                        f"{st.mean(x):.2f} ± {st.stdev(x):.2f}" if len(x) > 1 else f"{x[0]:.2f}" for x in v) + " |")
        print(("\nPRIMARY" if not kind else "\n(last)") + " E2R - E2, single seeds, s1_oov:")
        print("  diff %+.2f pp 95%% CI [%+.2f, %+.2f] p=%.4f" % boot(E["E2R", "single", "s1_oov"], E["E2", "single", "s1_oov"], rng))
        print("secondary E2R - E2 (single seeds):")
        for s in ("s1_test", "s2_test", "s2_oov"):
            print(f"  {s}: diff %+.2f [%+.2f, %+.2f] p=%.4f" % boot(E["E2R", "single", s], E["E2", "single", s], rng))
        if ("E2R", "soup", "s1_oov") in E:
            print("secondary E2R soup - E2 soup:")
            for s in ("s1_oov", "s1_test", "s2_oov", "s2_test"):
                print(f"  {s}: diff %+.2f [%+.2f, %+.2f] p=%.4f" % boot(E["E2R", "soup", s], E["E2", "soup", s], rng))


if __name__ == "__main__":
    main()
