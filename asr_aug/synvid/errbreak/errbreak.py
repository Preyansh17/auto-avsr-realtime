"""0-GPU breakdown of the deployed chain's 81 in-home errors (semg3_rcv_20261109_v1, 193 home_val clips) by cause.

Read-only on Tianyu's files. Hypotheses = deployed v1.8 chain (q8 fusion 0.1 + Qwen-0.5B re-rank, app45, top 10), rebuilt
exactly as pbias/step0_rescore.py. Every error token (substitution, deletion, insertion; total = edit distance) goes into
exactly one bucket, precedence fixed before counting:
  1 spoken  : the reference word (sub/del) or inserted word is a number / unit / "percent" word
  2 final   : not 1, and the token is the clip's last reference word (sub/del) or an insertion after it
  3 unseen  : not 1-2, sub/del of a reference word absent from the fold's training text (all rcv{k}_*_train roles +
              rcv{k}_real_all, i.e. what a fold model is trained on); split by whether the word occurs in the 917-row
              real-video pool (t2m2/pool/pool.jsonl)
  4 seen_sub: not 1-3, substitution of a seen word
  5 del_ins : not 1-4 (deletions of seen words, insertions)
Reach (20-best = first 20 distinct normalised candidates of the chain decode, the LM's app45 keys):
  sent20 = exact reference sentence among the 20 candidates; tok20 = token is correct in the 20-best oracle candidate
  (min edit distance, ties -> higher rank). Writes counts only (no transcripts) to --out.
usage: python3 errbreak.py --out /scratch/pa2753/synvid/errbreak/errbreak.json [--debug]
"""
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, "/scratch/pa2753/synvid/errbreak/code")
from step0_rescore import S, SR, K, norm, jl


# copied verbatim from pbias/step0_rescore.py (nested in main there)
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

NUM = set("""ZERO OH ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN
SEVENTEEN EIGHTEEN NINETEEN TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY HUNDRED THOUSAND MILLION BILLION POINT
FIRST SECOND THIRD FOURTH FIFTH SIXTH SEVENTH EIGHTH NINTH TENTH TWENTIETH HUNDREDTH DOLLAR DOLLARS CENT CENTS DEGREE
DEGREES PERCENT HALF QUARTER MINUTES HOURS O'CLOCK AM PM""".split())
isnum = lambda w: w in NUM or w.startswith("PERCENT") or any(c.isdigit() for c in w)


def ops(r, h):
    """Levenshtein alignment -> list of (op, ref_index or None, hyp_word or None); op in ok/sub/del/ins."""
    n, m = len(r), len(h)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            out.append(("ok" if r[i - 1] == h[j - 1] else "sub", i - 1, h[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out.append(("del", i - 1, None)); i -= 1
        else:
            out.append(("ins", i, h[j - 1])); j -= 1   # insertion before ref index i (i == n: after the last word)
    return out[::-1], D[n][m]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    G = {x["rel_path"]: norm(x["normalized_ground_truth"]) for x in json.load(open(S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json"))}
    fold_of, trainw = {}, {}
    for k in K:
        tw = set()
        for f in sorted(SR.glob(f"rcv{k}_*_train.jsonl")) + [SR / f"rcv{k}_real_all.jsonl"]:
            for x in jl(f):
                tw |= set(G[x["rel_path"]]) if x["rel_path"] in G else set(norm(x.get("raw_sentence") or x.get("transcript") or ""))
        trainw[k] = tw
        for x in jl(SR / f"rcv{k}_home_val.jsonl"): fold_of[x["rel_path"]] = k
    poolw = {w for l in open("/scratch/pa2753/synvid/t2m2/pool/pool.jsonl") for w in norm(json.loads(l)["norm"])}
    pred = S / "bp/chain/F1_app45_q8/fus_lm0.1_lb0/evaluation/dc_home_all.predictions.json"
    lmf = S / "lm/F1_chainq8_v3/Qwen2.5-0.5B.json"
    hyp = chain(pred, lmf)
    L = json.load(open(lmf))
    B = ["1_spoken", "2_final", "3a_unseen_no_video", "3b_unseen_has_video", "4_seen_sub", "5_del_ins"]
    rep = {b: dict(errors=0, clips=set(), tok20=0, sent20_errors=0, sent20_clips=set(), by_op=collections.Counter()) for b in B}
    total = 0; dbg = []; clip_sent20 = 0; clip_err = 0; pct_clips = 0
    for rp, k in fold_of.items():
        r, h = G[rp], hyp[rp]
        al, e = ops(r, h); total += e
        if not e: continue
        clip_err += 1
        cands = [c.upper().split() for c in list(L[rp]["app45"])[:20]]
        s20 = any(c == r for c in cands); clip_sent20 += s20
        best = min(cands, key=lambda c: ops(r, c)[1])
        bal, _ = ops(r, best)
        okref = {i for op, i, _ in bal if op == "ok"}
        oins = sum(op == "ins" for op, _, _ in bal)
        for op, i, hw in al:
            if op == "ok": continue
            w = r[i] if op != "ins" else hw
            if isnum(w) or (op == "ins" and hw and isnum(hw)): b = "1_spoken"
            elif (op != "ins" and i == len(r) - 1) or (op == "ins" and i == len(r)): b = "2_final"
            elif op != "ins" and w not in trainw[k]: b = "3b_unseen_has_video" if w in poolw else "3a_unseen_no_video"
            elif op == "sub": b = "4_seen_sub"
            else: b = "5_del_ins"
            R = rep[b]; R["errors"] += 1; R["clips"].add(rp); R["by_op"][op] += 1
            R["tok20"] += (i in okref) if op != "ins" else (oins == 0)
            if s20: R["sent20_errors"] += 1; R["sent20_clips"].add(rp)
            if a.debug: dbg.append((b, op, w, hw))
    out = dict(total_errors=total, n_clips=len(fold_of), clips_with_errors=clip_err, clips_with_errors_ref_in_20best=clip_sent20,
               pool_rows=sum(1 for _ in open("/scratch/pa2753/synvid/t2m2/pool/pool.jsonl")),
               precedence="1 spoken > 2 final > 3 unseen (3a no video / 3b has video) > 4 seen sub > 5 del/ins",
               buckets={b: dict(errors=v["errors"], clips=len(v["clips"]), by_op=dict(v["by_op"]), fixed_by_20best_oracle=v["tok20"],
                                errors_in_clips_with_ref_in_20best=v["sent20_errors"], clips_with_ref_in_20best=len(v["sent20_clips"]))
                        for b, v in rep.items()})
    assert sum(v["errors"] for v in out["buckets"].values()) == total
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    if a.debug:
        for d in sorted(dbg): print(*d, file=sys.stderr)


if __name__ == "__main__":
    main()
