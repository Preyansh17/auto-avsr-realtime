"""Step 2 scoring: per arm, last/first-word-wrong clips and total errors on dc_home_all (193) and dc_s3ov_val (35, per crop
variant); reproduction of arm a against F1 (in-home) and D0 (s3ov_val); paired bootstrap (10k, seed 0) of each arm vs a."""
import json, os, sys
sys.path.insert(0, "/scratch/pa2753/synvid/endclip")
from endlib import clip_stats, summarize, paired_bootstrap

E = "/scratch/pa2753/synvid/endclip"
SETS = {"dc_home_all": ("/scratch/th3482/Dysarthric_ASR/runs/s3v34/home_eval_v1/bp/decgrid/F1/c0.1_b10/evaluation/dc_home_all.predictions.json",
                        {a: f"{E}/dec/{a}" for a in "abc"})}
for v in ("fi_v1", "fu_v1", "fv_v1"):
    SETS[f"dc_s3ov_val@{v}"] = ("/scratch/th3482/Dysarthric_ASR/runs/s3v34/runs/d0/deliv_v/evaluation/dc_s3ov_val.predictions.json",
                                {a: f"{E}/dec_ov/{v}/{a}" for a in "abc"})

def load(p):
    return {x["rel_path"]: x for x in json.load(open(p))}

res = {}
for name, (refp, dirs) in SETS.items():
    role = name.split("@")[0]; ref = load(refp); order = sorted(ref)
    P = {"ref": ref}
    for a, d in dirs.items():
        f = f"{d}/evaluation/{role}.predictions.json"
        if os.path.exists(f): P[a] = load(f)
    S = {k: [clip_stats(p[c]["ground_truth"], p[c]["prediction"]) for c in order] for k, p in P.items()}
    out = {k: summarize(v) for k, v in S.items()}
    if "a" in P:
        out["repro_a"] = dict(clips_text_differ=sum(P["a"][c]["prediction"] != ref[c]["prediction"] for c in order),
                              err_a_minus_ref=out["a"]["err"] - out["ref"]["err"])
        for a in "bc":
            if a in S:
                out[f"boot_{a}_vs_a"] = paired_bootstrap(S["a"], S[a], keys=("last_wrong", "first_wrong", "err"))
    res[name] = out
json.dump(res, open(f"{E}/scores.json", "w"), indent=1)
print(json.dumps(res, indent=1))
