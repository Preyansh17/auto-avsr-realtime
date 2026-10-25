"""Section H analysis: per-arm WER and paired two-level bootstrap (seeds, then sentences), Holm over 3 pairs x {test,oov}.
O/OS have seeds 1-5; N, NS, ON, ONS have seeds 1-10 (pairs use the seeds both arms share)."""
import csv, random, re, statistics as st
R = "/scratch/pa2753/asr_aug/runs"
def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
def load(arm, s, split, kind):
    f = f"{R}/oct1_{arm}_seed{s}/{split}_hyps{kind}.tsv"
    rows = list(csv.reader(open(f), delimiter="\t"))[1:]
    return [(ed(norm(r[1]).split(), norm(r[2]).split()), len(norm(r[1]).split())) for r in rows]
SEEDS = {"O": range(1, 6), "OS": range(1, 6), "N": range(1, 11), "NS": range(1, 11), "ON": range(1, 11), "ONS": range(1, 11)}
ARMS = list(SEEDS)
for kind in ["", "_last"]:
    print("==", "best(val-selected)" if not kind else "last epoch")
    D = {(a, s, sp): load(a, s, sp, kind) for a in ARMS for s in SEEDS[a] for sp in ("val", "test", "oov")}
    wer = lambda r: 100 * sum(e for e, _ in r) / sum(n for _, n in r)
    for sp in ("val", "test", "oov"):
        print(sp, "  ".join(f"{a} {st.mean(wer(D[a,s,sp]) for s in SEEDS[a]):.1f}±{st.stdev(wer(D[a,s,sp]) for s in SEEDS[a]):.1f}" for a in ARMS))
    rng = random.Random(0); res = []
    for base, syn in [("O", "OS"), ("N", "NS"), ("ON", "ONS")]:
        for sp in ("test", "oov"):
            S = SEEDS[base]; n = len(D[base, 1, sp]); bs = []
            for _ in range(4000):
                ss = [rng.choice(list(S)) for _ in S]; idx = [rng.randrange(n) for _ in range(n)]
                f = lambda a: 100 * sum(sum(D[a, s, sp][i][0] for i in idx) for s in ss) / sum(sum(D[a, s, sp][i][1] for i in idx) for s in ss)
                bs.append(f(syn) - f(base))
            bs.sort(); p = 2 * min(sum(b >= 0 for b in bs), sum(b <= 0 for b in bs)) / len(bs)
            d = st.mean(wer(D[syn, s, sp]) for s in S) - st.mean(wer(D[base, s, sp]) for s in S)
            res.append((p, f"{syn}-{base} {sp} (n_seeds={len(S)}): diff {d:+.2f}pp 95%CI [{bs[100]:+.2f},{bs[3900]:+.2f}] p={p:.3f}"))
    for i, (p, t) in enumerate(sorted(res)): print(t, f"Holm p={min(1, p * (len(res) - i)):.3f}")
