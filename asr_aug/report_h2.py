"""PLAN.md section H2: 2-fold oov-text synthetic. NF/ONF trained on fold X text are scored only on oov rows whose
sentence is in the other fold; fold-pooled per seed. Comparators N, ON, NS, ONS (seeds 1-5) scored on the same rows.
Paired two-level bootstrap (seeds, sentences); primary NF-N and ONF-ON, Holm over those 2; secondary vs NS/ONS."""
import csv, json, random, re, statistics as st
R = "/scratch/pa2753/asr_aug/runs"; DATA = "/scratch/pa2753/asr_aug/data"
def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
FOLD = {F: {e["norm"] for e in json.load(open(f"{DATA}/sents_fold{F}.json"))} for F in "AB"}
S = range(1, 6)
def rows(run, kind):
    out = []
    for r in list(csv.reader(open(f"{R}/{run}/oov_hyps{kind}.tsv"), delimiter="\t"))[1:]:
        ref = norm(r[1]); out.append((ref, ed(ref.split(), norm(r[2]).split()), len(ref.split())))
    return out
def get(arm, s, kind):
    """Rows (sentence-ordered) scored for this arm/seed. F arms: fold-pooled, other-fold rows only."""
    if arm.endswith("F"):
        res = []
        for F, O in (("A", "B"), ("B", "A")):
            res += [(ref, e, n) for ref, e, n in rows(f"oct1_{arm}_fold{F}_seed{s}", kind) if ref in FOLD[O]]
        return res
    return [x for x in rows(f"oct1_{arm}_seed{s}", kind) if x[0] in FOLD["A"] | FOLD["B"]]
def run(kind):
    print("==", "best(val-selected)" if not kind else "last epoch")
    D = {(a, s): get(a, s, kind) for a in ("N", "NF", "NS", "ON", "ONF", "ONS") for s in S}
    wer = lambda r: 100 * sum(e for _, e, _ in r) / sum(n for _, _, n in r)
    for a in ("N", "NF", "NS", "ON", "ONF", "ONS"):
        v = [wer(D[a, s]) for s in S]; print(f"{a} oov(fold-scored) {st.mean(v):.1f}±{st.stdev(v):.1f}")
    # bootstrap over unique sentences, grouping rows by sentence so arms with different row sets stay paired
    sents = sorted({x[0] for a in D for x in D[a, 1]})
    def agg(a, s, ss):
        idx = {}
        for ref, e, n in D[a, s]: idx.setdefault(ref, []).append((e, n))
        return idx
    A = {(a, s): agg(a, s, None) for (a, s) in D}
    rng = random.Random(0); res = []
    for syn, base, tag in [("NF", "N", "P"), ("ONF", "ON", "P"), ("NF", "NS", "S"), ("ONF", "ONS", "S")]:
        bs = []
        for _ in range(4000):
            ss = [rng.choice(list(S)) for _ in S]; pick = [rng.choice(sents) for _ in sents]
            def f(a):
                e = n = 0
                for s in ss:
                    for q in pick:
                        for x, y in A[a, s].get(q, []): e += x; n += y
                return 100 * e / n
            bs.append(f(syn) - f(base))
        bs.sort(); p = 2 * min(sum(b >= 0 for b in bs), sum(b <= 0 for b in bs)) / len(bs)
        d = st.mean(wer(D[syn, s]) for s in S) - st.mean(wer(D[base, s]) for s in S)
        res.append((tag, p, f"{syn}-{base}: diff {d:+.2f}pp 95%CI [{bs[100]:+.2f},{bs[3900]:+.2f}] p={p:.4f}"))
    prim = sorted(r for r in res if r[0] == "P")
    for i, (_, p, t) in enumerate(prim): print("PRIMARY", t, f"Holm p={min(1, p * (len(prim) - i)):.4f}")
    for tag, p, t in res:
        if tag == "S": print("secondary", t)
for k in ["", "_last"]: run(k)
