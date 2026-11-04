# Descriptive follow-ups: seed-paired CI (same seed in TTA and step 0) and how many hypotheses TTA changed.
import random, statistics as st, sys
sys.path.insert(0, "/scratch/pa2753/stack/code")
from report_stack import R, SETS, errs, hyps, refs, wer
ref = {s: refs(s) for s in SETS}; ref["s2_eval"] = {**ref["s2_test"], **ref["s2_oov"]}
def H(arm, x, sc, k, s):
    if s == "s2_eval": return {**H(arm, x, sc, k, "s2_test"), **H(arm, x, sc, k, "s2_oov")}
    return hyps(f"{R}/{arm}_seed{x}/tta/{sc}/step{k}_{s}_hyps.tsv")
def paired(arm, sc, k, s, rng, n=4000):
    A = [errs(ref[s], H(arm, x, sc, k, s)) for x in range(1, 6)]; B = [errs(ref[s], H(arm, x, sc, 0, s)) for x in range(1, 6)]
    by = {}
    for t, (r, _, _) in A[0].items(): by.setdefault(r, []).append(t)
    sents = sorted(by); d0 = st.mean(wer(a) - wer(b) for a, b in zip(A, B)); bs = []
    for _ in range(n):
        pick = [t for q in (rng.choice(sents) for _ in sents) for t in by[q]]
        idx = [rng.randrange(5) for _ in range(5)]
        w = lambda e: 100 * sum(e[t][1] for t in pick) / sum(e[t][2] for t in pick)
        bs.append(st.mean(w(A[i]) - w(B[i]) for i in idx))
    bs.sort(); return d0, bs[int(.025 * n)], bs[int(.975 * n)]
rng = random.Random(0)
print("seed-paired CI, TTA - step 0")
for name, arm, sc, k, s in (("selected", "E2", "ln", 1, "s1_oov"), ("selected", "E1", "ln", 1, "s2_eval")) + tuple(("step10", a, c, 10, s) for a, s in (("E2", "s1_oov"), ("E1", "s2_eval")) for c in ("ln", "feat")):
    print(f"  {name} {arm} {sc} {k} {s}: %+.2f [%+.2f, %+.2f]" % paired(arm, sc, k, s, rng))
print("hypotheses changed from step 0 (of all utterances x 5 seeds)")
for arm in ("E1", "E2"):
    for sc in ("ln", "feat"):
        out = []
        for k in (1, 3, 5, 10):
            c = n = 0
            for x in range(1, 6):
                for s in SETS:
                    a, b = H(arm, x, sc, k, s), H(arm, x, sc, 0, s)
                    c += sum(a[t] != b[t] for t in a); n += len(a)
            out.append(f"step{k} {c}/{n} ({100*c/n:.1f}%)")
        print(f"  {arm} {sc}: " + ", ".join(out))
