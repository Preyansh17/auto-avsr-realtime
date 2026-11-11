"""Step 0 (read-only): rescore Tianyu's existing pvocab / chain predictions on the semg3_rcv_20261109_v1 5-fold in-home CV.

Reads only. Writes one JSON summary to --out (under /scratch/pa2753/synvid/pbias/). Reports per arm:
pooled errors/words on the 193 home_val clips, errors on seen vs unseen reference words (seen = word occurs in that clip's fold
home_train transcripts), "percent" clips fixed, last-word errors, leave-one-fold-out strength pick, paired bootstrap vs plain.
Also checks the pvocab lists for leakage (every list word must occur in the fold's home_train; how many also occur in home_val).
usage: python3 step0_rescore.py --out step0.json
"""
import argparse, json, random, re
from pathlib import Path

S = Path("/scratch/th3482/Dysarthric_ASR/runs/s3v34/home_eval_v1")
SR = Path("/scratch/th3482/Dysarthric_ASR/data/patients/splits/semg3_rcv_20261109_v1/roles")
K = range(5)


def norm(s):
    return re.sub(r"[^A-Z0-9' ]", " ", s.upper()).split()


def align(r, h):
    """Levenshtein; returns (edits, per-ref-word error flags, insertions)."""
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, bad, ins = n, m, [False] * n, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            bad[i - 1] = r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            bad[i - 1] = True; i -= 1
        else:
            ins += 1; j -= 1
    return D[n][m], bad, ins


def jl(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    res = {}
    # references: the eval's own normalised ground truth (numbers spelled out), same for every arm
    G = {x["rel_path"]: norm(x["normalized_ground_truth"]) for x in json.load(open(S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json"))}
    fold_of, ref_of, trainw = {}, {}, {}
    val_ids = {k: set() for k in K}
    for k in K:
        tr = jl(SR / f"rcv{k}_home_train.jsonl"); va = jl(SR / f"rcv{k}_home_val.jsonl")
        trainw[k] = {w for x in tr for w in G[x["rel_path"]]}
        tr_ids = {x["rel_path"] for x in tr}
        for x in va:
            fold_of[x["rel_path"]] = k; ref_of[x["rel_path"]] = G[x["rel_path"]]; val_ids[k].add(x["rel_path"])
        assert not (tr_ids & val_ids[k]), f"fold {k}: train/val overlap"
    res["home_train_sizes"] = {k: len(jl(SR / f"rcv{k}_home_train.jsonl")) for k in K}
    res.update(n_clips=len(fold_of), n_words=sum(len(v) for v in ref_of.values()))

    # leakage check of pvocab lists
    leak = {}
    for k in K:
        lst = [w.strip().upper() for w in open(S / f"pvocab_rcv_v1/f{k}.txt") if w.strip()]
        valw = {w for rp in val_ids[k] for w in ref_of[rp]}
        leak[k] = dict(n=len(lst), not_in_train=[w for w in lst if w not in trainw[k]], also_in_val=[w for w in lst if w in valw],
                       has_percent=[w for w in lst if w.startswith("PERCENT")])
    res["pvocab_lists"] = leak

    def load(paths):
        h = {}
        for p in paths:
            for x in json.load(open(p)):
                h[x["rel_path"]] = norm(x["normalized_prediction"])
        return h

    def chain(pred, lmf, tag="app45", topk=10, lam=0.5, mu=0.0):
        """v1.8 deployed re-rank: vsr + lam * Qwen lm(c | app45 ctx) + mu * n_words over the first topk distinct candidates."""
        tnorm = lambda t: " ".join(t.lower().split())  # model candidates are upper-case word pieces; misses are counted
        L = json.load(open(lmf)); h = {}; miss = 0
        for x in json.load(open(pred)):
            nb, sc = x["nbest"], x["nbest_scores"]
            if isinstance(nb, str): nb, sc = eval(nb), eval(sc)
            # map raw candidates onto the LM's keys (keys = distinct normalised candidates in n-best order): exact lower-case
            # match if present, else the next unused key in order (the normaliser expands contractions / numbers)
            lmd = L[x["rel_path"]][tag]; keys = list(lmd); used = set(); key_of = []
            for t in nb:
                k = tnorm(t)
                if k not in lmd:
                    nxt = [q for q in keys if q not in used and q not in {tnorm(u) for u in nb}]
                    k = nxt[0] if nxt else k
                key_of.append(k); used.add(k)
            c = {}
            for k, s_ in zip(key_of, sc):
                if k and k not in c and len(c) < topk: c[k] = float(s_)
            lmd = L[x["rel_path"]][tag]; miss += sum(k not in lmd for k in c)
            h[x["rel_path"]] = max(c, key=lambda k: c[k] + lam * lmd.get(k, -1e9) + mu * len(k.split())).upper().split()
        print("chain lm misses", miss)
        return h

    arms = {f"pvocab_b{b}": load([S / f"bp/pvocab_rcv/F1/b{b}/f{k}/evaluation/rcv{k}_home_val.predictions.json" for k in K]) for b in range(4)}
    for c in ("F1_app45_fp32", "F1_app45_q8"):
        for f in ("fus_lm0_lb0", "fus_lm0.1_lb0"):
            p = S / f"bp/chain/{c}/{f}/evaluation/dc_home_all.predictions.json"
            if p.exists(): arms[f"chain_{c}_{f}"] = load([p])
    for c, lmd in (("F1_app45_q8", "F1_chainq8_v3"),):
        arms[f"chain_rerank_{c}"] = chain(S / f"bp/chain/{c}/fus_lm0.1_lb0/evaluation/dc_home_all.predictions.json",
                                          S / f"lm/{lmd}/Qwen2.5-0.5B.json")
    p = S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json"
    if p.exists(): arms["decgrid_F1_c0.1_b10"] = load([p])

    pct = [rp for rp, r in ref_of.items() if any(w.startswith("PERCENT") for w in r)]
    res["percent_clips"] = {rp: " ".join(ref_of[rp]) for rp in pct}
    per = {}
    for name, hyp in arms.items():
        miss = [rp for rp in ref_of if rp not in hyp]
        E = seenE = unseenE = seenN = unseenN = last = ins = 0; pc = []; clip = {}
        for rp, r in ref_of.items():
            h = hyp.get(rp, [])
            e, bad, i_ = align(r, h); E += e; ins += i_; clip[rp] = e
            for w, b in zip(r, bad):
                if w in trainw[fold_of[rp]]: seenN += 1; seenE += b
                else: unseenN += 1; unseenE += b
            last += bool(r) and bad[-1]
            if rp in pct: pc.append(dict(ref=" ".join(r), hyp=" ".join(h), edits=e))
        per[name] = clip
        res[name] = dict(missing=len(miss), errors=E, wer=round(100 * E / res["n_words"], 2), insertions=ins,
                         seen_err=seenE, seen_n=seenN, unseen_err=unseenE, unseen_n=unseenN, last_word_err=last,
                         percent_clips_correct=sum(p["edits"] == 0 for p in pc), percent=pc,
                         per_fold={k: sum(clip[rp] for rp in val_ids[k]) for k in K})

    # leave-one-fold-out pick of pvocab strength among b1..b3, applied fold by fold
    lofo, Elofo, pick = {}, 0, {}
    for k in K:
        others = [j for j in K if j != k]
        b = min((1, 2, 3), key=lambda b: (sum(res[f"pvocab_b{b}"]["per_fold"][j] for j in others), b))
        pick[k] = b
        for rp in val_ids[k]: lofo[rp] = per[f"pvocab_b{b}"][rp]
    per["pvocab_lofo"] = lofo
    res["pvocab_lofo"] = dict(pick=pick, errors=sum(lofo.values()))

    # paired bootstrap over clips vs plain (b0)
    rng = random.Random(0); ids = list(ref_of); base = per["pvocab_b0"]; boot = {}
    for name in [n for n in per if n != "pvocab_b0"]:
        d = [per[name][rp] - base[rp] for rp in ids]; s = []
        for _ in range(10000):
            s.append(sum(d[rng.randrange(len(d))] for _ in ids))
        s.sort(); boot[name] = dict(delta=sum(d), ci95=[s[249], s[9749]])
    res["bootstrap_vs_b0"] = boot
    a.out.write_text(json.dumps(res, indent=1))
    for n in per:
        r = res.get(n, {}); print(n, r.get("errors"), r.get("unseen_err"), r.get("percent_clips_correct"), boot.get(n))


if __name__ == "__main__":
    main()
