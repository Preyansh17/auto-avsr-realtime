"""Score rh4sp (and other already-decoded fold models) through the deployed v1.8 re-rank on the 193 in-home clips.
CPU only: reuses Tianyu's existing chain decodes (v/fus_lm0.1_lb0, app45 ctx, int8+fp16) and Qwen2.5-0.5B scores; the
re-rank (vsr + 0.5 * lm, top 10, app45) is errbreak.chain, identical to the deployed F1 chain baseline (81 errors).
Paired clip bootstrap (4,000 draws, seed 20261115) vs the F1 chain. Counts only."""
import glob, json, random, sys
sys.path.insert(0, "/scratch/pa2753/synvid/errbreak/code")
from errbreak import chain, ops, S, SR, K, norm, jl
G = {x["rel_path"]: norm(x["normalized_ground_truth"]) for x in json.load(open(S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json"))}
P = {rp for rp, r in G.items() if "PERCENT" in r}
base = chain(S / "bp/chain/F1_app45_q8/fus_lm0.1_lb0/evaluation/dc_home_all.predictions.json", S / "lm/F1_chainq8_v3/Qwen2.5-0.5B.json")
E0 = {rp: ops(G[rp], base[rp])[1] for rp in G}
R = "/scratch/th3482/Dysarthric_ASR/runs/s3v34"
cands = {"rh4sp_soup4 (primary)": R + "/runs/tw1/rh4sp/f{k}/soup4_a1.0/v"}
for d in sorted(glob.glob(R + "/home_eval_v1/bp/*chain*/*/f0/v")):
    cands[d.split("/bp/")[1].replace("/f0/v", "")] = d.replace("/f0/", "/f{k}/")
out = {"F1_chain": dict(errors=sum(E0.values()), percent_exact=sum(base[rp] == G[rp] for rp in P))}
for name, pat in cands.items():
    h = {}
    try:
        for k in K:
            h.update(chain(f"{pat.format(k=k)}/fus_lm0.1_lb0/evaluation/rcv{k}_home_val.predictions.json", f"{pat.format(k=k)}/lm_Qwen2.5-0.5B.json"))
    except (FileNotFoundError, KeyError) as e:
        out[name] = f"skip: {type(e).__name__}"; continue
    if set(h) != set(G): out[name] = f"skip: {len(h)} clips"; continue
    E = {rp: ops(G[rp], h[rp])[1] for rp in G}; clips = sorted(G); d = [E[c] - E0[c] for c in clips]
    rng = random.Random(20261115); b = sorted(sum(d[rng.randrange(len(d))] for _ in d) for _ in range(4000))
    out[name] = dict(errors=sum(E.values()), diff=sum(d), ci95=[b[100], b[3899]], fixed=sum(x < 0 for x in d), broken=sum(x > 0 for x in d),
                     percent_exact=sum(h[rp] == G[rp] for rp in P))
print(json.dumps(out, indent=1)); open("/scratch/pa2753/synvid/errbreak/rh4chain.json", "w").write(json.dumps(out, indent=1) + "\n")
