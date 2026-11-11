"""pbias step 1: n-best phrase-bias re-ranking on the semg3 random 5-fold in-home CV (pre-registered in README.md).

CPU only, reads Tianyu's existing n-best predictions (read-only), writes JSON to --out. No decoding.
usage: python3 pbias.py --out step1.json
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

S = Path("/scratch/th3482/Dysarthric_ASR/runs/s3v34/home_eval_v1")
SR = Path("/scratch/th3482/Dysarthric_ASR/data/patients/splits/semg3_rcv_20261109_v1/roles")
K = range(5)
STRENGTHS = (0.5, 1, 2)
STOP = set("""a an the and or but if so of to in on at by for with from up down out over under into onto off as than then
i me my mine myself you your yours we us our ours he him his she her hers it its they them their theirs this that these those
is am are was were be been being have has had do does did done will would shall should can could may might must not no yes
yeah okay ok oh um uh what which who whom whose when where why how all any some each just very too also there here now
let let's go going get got say said see like know think want well one""".split())


def W(t):
    return re.sub(r"[^a-z0-9' ]", " ", t.lower()).split()


def jl(p):
    return [json.loads(l) for l in open(p)]


def build_list(texts):
    words = {w for t in texts for w in t if w not in STOP}
    df = Counter()
    for t in texts:
        df.update({tuple(t[i:i + n]) for n in (2, 3, 4) for i in range(len(t) - n + 1)})
    phrases = {g for g, c in df.items() if c >= 2 and any(w not in STOP for w in g)}
    return words, phrases


def nbias(c, words, phrases):
    t = c.split()
    return sum(w in words for w in t) + sum(tuple(t[i:i + len(g)]) == g for g in phrases for i in range(len(t) - len(g) + 1))


def align(r, h):
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, bad = n, m, [False] * n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            bad[i - 1] = r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            bad[i - 1] = True; i -= 1
        else:
            j -= 1
    return D[n][m], bad


def pools_plain(files, topk=20):
    """{rel_path: (ref_words, [(cand, base_score)])} from usr2_eval predictions; candidates normalised by lower-casing."""
    out = {}
    for f in files:
        for x in json.load(open(f)):
            nb, sc = x["nbest"], x["nbest_scores"]
            if isinstance(nb, str): nb, sc = eval(nb), eval(sc)
            c = {}
            for t, s in zip(nb, sc):
                k = " ".join(W(t))
                if k and k not in c and len(c) < topk: c[k] = float(s)
            out[x["rel_path"]] = (W(x["normalized_ground_truth"]), list(c.items()))
    return out


def pools_chain(pred, lmf, tag="app45", topk=10, lam=0.5, mu=0.0):
    """Deployed v1.8 re-rank pool: base = vsr + lam * lm(c | app45) + mu * n_words over the first topk distinct candidates.
    Raw candidates are mapped onto the LM's keys (distinct normalised candidates in n-best order)."""
    L = json.load(open(lmf)); out = {}; miss = 0
    for x in json.load(open(pred)):
        nb, sc = x["nbest"], x["nbest_scores"]
        if isinstance(nb, str): nb, sc = eval(nb), eval(sc)
        lmd = L[x["rel_path"]][tag]; keys = list(lmd); low = {" ".join(t.lower().split()) for t in nb}; key_of = []
        for t in nb:
            k = " ".join(t.lower().split())
            if k not in lmd:
                nxt = [q for q in keys if q not in key_of and q not in low]
                k = nxt[0] if nxt else k
            key_of.append(k)
        c = {}
        for k, s in zip(key_of, sc):
            if k and k not in c and len(c) < topk:
                miss += k not in lmd; c[k] = float(s) + lam * lmd.get(k, -1e9) + mu * len(k.split())
        out[x["rel_path"]] = (W(x["normalized_ground_truth"]), [(" ".join(W(k)), v) for k, v in c.items()])
    print("chain lm misses", miss)
    return out


def pick(cands, s, words, phrases):
    return max(enumerate(cands), key=lambda ic: (ic[1][1] + s * nbias(ic[1][0], words, phrases), -ic[0]))[1][0].split()


def boot(d, B=10000, seed=0):
    rng = random.Random(seed); n = len(d); v = []
    for _ in range(B):
        v.append(sum(d[rng.randrange(n)] for _ in range(n)))
    v.sort(); return [v[int(.025 * B)], v[int(.975 * B) - 1]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    G = {x["rel_path"]: W(x["normalized_ground_truth"]) for x in json.load(open(S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json"))}
    val, lists, trainw = {}, {}, {}
    for k in K:
        tr = [G[x["rel_path"]] for x in jl(SR / f"rcv{k}_home_train.jsonl")]
        va = [x["rel_path"] for x in jl(SR / f"rcv{k}_home_val.jsonl")]
        assert not set(va) & {x["rel_path"] for x in jl(SR / f"rcv{k}_home_train.jsonl")}
        for rp in va: val[rp] = k
        lists[k] = build_list(tr); trainw[k] = {w for t in tr for w in t}
    res = {"lists": {k: dict(n_words=len(w), n_phrases=len(p), has_percent="percent" in w,
                              numbers=sorted(x for x in w if x in {"ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                                                                 "eighty", "ninety", "hundred", "five", "eight"}))
                     for k, (w, p) in lists.items()}}
    models = {
        # F1 plain decode (= pvocab b0 = decgrid_F1 c0.1_b10, 112/1116); candidates keyed by the eval normaliser via the LM file
        "F1": pools_chain(S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json",
                          S / "lm/F1_c01b10_v3/Qwen2.5-0.5B.json", topk=20, lam=0.0),
        "chain": pools_chain(S / "bp/chain/F1_app45_q8/fus_lm0.1_lb0/evaluation/dc_home_all.predictions.json",
                             S / "lm/F1_chainq8_v3/Qwen2.5-0.5B.json"),
    }
    pct = sorted(rp for rp in val if "percent" in G[rp])
    for mname, P in models.items():
        P = {rp: P[rp] for rp in val}
        assert len(P) == 193
        hyps = {s: {rp: pick(P[rp][1], s, *lists[val[rp]]) for rp in P} for s in (0,) + STRENGTHS}
        err = {s: {rp: align(G[rp], h[rp])[0] for rp in P} for s, h in hyps.items()}
        lofo, pk = {}, {}
        for k in K:
            s_ = min(STRENGTHS, key=lambda s: (sum(e for rp, e in err[s].items() if val[rp] != k), s)); pk[k] = s_
            for rp in P:
                if val[rp] == k: lofo[rp] = s_
        arms = {"none": {rp: hyps[0][rp] for rp in P}, "lofo": {rp: hyps[lofo[rp]][rp] for rp in P}}
        arms.update({f"s{s}": hyps[s] for s in STRENGTHS})
        out = {"lofo_pick": pk, "oracle_in_pool": sum(" ".join(G[rp]) in [c for c, _ in P[rp][1]] for rp in P)}
        base = {rp: align(G[rp], arms["none"][rp])[0] for rp in P}
        for an, H in arms.items():
            E = se = un = last = 0; per = {}; pc = []
            for rp in P:
                r = G[rp]; e, bad = align(r, H[rp]); per[rp] = e; E += e; last += bool(r) and bad[-1]
                for w, b in zip(r, bad):
                    if w in trainw[val[rp]]: se += b
                    else: un += b
                if rp in pct: pc.append(dict(ref=" ".join(r), hyp=" ".join(H[rp]), edits=e, base_edits=base[rp]))
            d = [per[rp] - base[rp] for rp in sorted(P)]
            out[an] = dict(errors=E, wer=round(100 * E / 1116, 2), seen_err=se, unseen_err=un, last_word_err=last,
                           percent_exact=sum(p["edits"] == 0 for p in pc), percent_fixed=sum(p["edits"] == 0 < p["base_edits"] for p in pc),
                           percent_broken=sum(p["edits"] > 0 == p["base_edits"] for p in pc),
                           clips_fixed=sum(per[rp] < base[rp] for rp in P), clips_broken=sum(per[rp] > base[rp] for rp in P),
                           delta=sum(d), ci95=boot(d) if an != "none" else [0, 0], percent=pc)
            print(mname, an, E, "unseen", un, "pct", out[an]["percent_exact"], "/", len(pc), "d", sum(d), out[an]["ci95"], flush=True)
        res[mname] = out
    res["n_percent_clips"] = len(pct)
    res["seen_n"] = sum(w in trainw[val[rp]] for rp in val for w in G[rp]); res["unseen_n"] = 1116 - res["seen_n"]

    # do-no-harm: F1 alone, deployment list from all 193 in-home transcripts, on scripted val sets (no in-home clips)
    wd, pd = build_list([G[rp] for rp in val]); dnh = {}
    for name, roles in (("v34_val", ["dc_v34_val"]), ("s3_val+s3ov_val", ["dc_s3_val", "dc_s3ov_val"])):
        P = pools_plain([S / f"bp/decgrid_F1/c0.1_b10/evaluation/{r}.predictions.json" for r in roles])
        P = {rp: v for rp, v in P.items() if rp not in G}
        N = sum(len(v[0]) for v in P.values())
        ev = sum(int(x["edit_distance"]) for r in roles for x in json.load(open(S / f"bp/decgrid_F1/c0.1_b10/evaluation/{r}.predictions.json"))
                 if x["rel_path"] not in G)
        dnh[name] = dict(n_clips=len(P), n_words=N, eval_errors=ev, **{f"s{s}": sum(align(P[rp][0], pick(P[rp][1], s, wd, pd))[0] for rp in P)
                                                       for s in (0,) + STRENGTHS})
        print("dnh", name, dnh[name], flush=True)
    res["do_no_harm_F1"] = dnh
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
