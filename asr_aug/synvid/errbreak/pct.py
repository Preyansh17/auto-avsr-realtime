import json, sys
from step0_rescore import S, SR, K, norm, jl
G = {x["rel_path"]: norm(x["normalized_ground_truth"]) for x in json.load(open(S / "bp/decgrid_F1/c0.1_b10/evaluation/dc_home_all.predictions.json"))}
P = {rp for rp, r in G.items() if "PERCENT" in r}
fold = {x["rel_path"]: k for k in K for x in jl(SR / f"rcv{k}_home_val.jsonl")}
D = "/scratch/th3482/Dysarthric_ASR/runs/s3v34/runs/tw1/rh4sp"
out = {"percent_clips": len(P)}
for tag in ["seed1", "seed3", "seed7", "seed42", "soup4_a1.0/plain", "soup4_a1.0/v_b20/fus_lm0.1_lb0"]:
    has = exact = 0; n = 0
    for k in K:
        sub = tag if tag.startswith("soup") else f"{tag}/a1.0"
        for x in json.load(open(f"{D}/f{k}/{sub}/evaluation/rcv{k}_home_val.predictions.json")):
            if x["rel_path"] in P:
                n += 1; h = norm(x["normalized_prediction"]); has += "PERCENT" in h; exact += h == G[x["rel_path"]]
    out[tag] = dict(n=n, hyp_has_percent=has, exact=exact)
print(json.dumps(out))
