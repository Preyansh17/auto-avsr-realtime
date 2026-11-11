"""Spoken-form "percent" post-rule on the deployed v1.8 chain output. 0 GPU, CPU only. Rule frozen before scoring (2026-11-11):
R1 (primary): if the chain output's last word is a number word (errbreak.isnum, not PERCENT) and one of the first 20
    distinct n-best candidates equals the output + " PERCENT", output that candidate.
R2 (descriptive, looser): if the output's last word is a number word and any of the 20 candidates contains PERCENT,
    append PERCENT to the output.
CAVEAT: the rule was designed after seeing the in-home errors (errbreak), so the in-home gain is in-sample.
Do-no-harm: apply R1/R2 to F1 plain-beam outputs (top-1) of five non-home eval sets, which have no app context
(dc_s3_val, dc_s3ov_val, dc_v34_val, dc_s3_test, dc_s3_oov); report fires and errors before/after.
Paired clip bootstrap vs chain (4,000 draws, seed 20261115). Counts only."""
import json, random, sys
sys.path.insert(0, "/scratch/pa2753/synvid/errbreak/code")
from errbreak import chain, ops, isnum, S, norm

def cands(x, n=20):
    nb = x["nbest"]; nb = eval(nb) if isinstance(nb, str) else nb; out = []
    for t in nb:
        t = norm(t)
        if t and t not in out: out.append(t)
    return out[:n]

def rule(h, cs, r):
    if not h or not isnum(h[-1]) or h[-1].startswith("PERCENT"): return h, False
    if r == "R1":
        t = h + ["PERCENT"]
        return (t, True) if t in cs else (h, False)
    return (h + ["PERCENT"], True) if any("PERCENT" in w for c in cs for w in c) else (h, False)

PF = S / "bp/chain/F1_app45_q8/fus_lm0.1_lb0/evaluation/dc_home_all.predictions.json"
X = {x["rel_path"]: x for x in json.load(open(PF))}
G = {rp: norm(x["normalized_ground_truth"]) for rp, x in X.items()}
base = chain(PF, S / "lm/F1_chainq8_v3/Qwen2.5-0.5B.json")
E0 = {rp: ops(G[rp], base[rp])[1] for rp in G}; clips = sorted(G); P = {rp for rp in G if "PERCENT" in G[rp]}
out = {"home_chain": dict(errors=sum(E0.values()), percent_exact=sum(base[rp] == G[rp] for rp in P))}
for r in ("R1", "R2"):
    h, fire = {}, set()
    for rp in clips:
        h[rp], f = rule(base[rp], cands(X[rp]), r)
        if f: fire.add(rp)
    E = {rp: ops(G[rp], h[rp])[1] for rp in clips}; d = [E[c] - E0[c] for c in clips]
    rng = random.Random(20261115); b = sorted(sum(d[rng.randrange(len(d))] for _ in d) for _ in range(4000))
    out[f"home_{r}"] = dict(errors=sum(E.values()), diff=sum(d), ci95=[b[100], b[3899]], fires=len(fire),
                            fires_on_percent_ref=len(fire & P), fixed=sum(x < 0 for x in d), broken=sum(x > 0 for x in d),
                            percent_exact=sum(h[rp] == G[rp] for rp in P))
for role in ("dc_s3_val", "dc_s3ov_val", "dc_v34_val", "dc_s3_test", "dc_s3_oov"):
    xs = json.load(open(S / f"bp/decgrid_F1/c0.1_b10/evaluation/{role}.predictions.json"))
    rec = dict(clips=len(xs), refs_with_percent=sum("PERCENT" in norm(x["normalized_ground_truth"]) for x in xs), refs_ending_number=sum(bool(norm(x["normalized_ground_truth"])) and isnum(norm(x["normalized_ground_truth"])[-1]) for x in xs))
    for r in ("R1", "R2"):
        e0 = e1 = fires = 0
        for x in xs:
            g = norm(x["normalized_ground_truth"]); h0 = norm(x["normalized_prediction"])
            h1, f = rule(h0, cands(x), r); fires += f
            e0 += ops(g, h0)[1]; e1 += ops(g, h1)[1]
        rec[r] = dict(fires=fires, errors_before=e0, errors_after=e1)
    out[role] = rec
print(json.dumps(out, indent=1)); open("/scratch/pa2753/synvid/errbreak/pctrule.json", "w").write(json.dumps(out, indent=1) + "\n")
