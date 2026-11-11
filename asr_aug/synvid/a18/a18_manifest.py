"""A18: training manifest for arm A3x5 = V3's 169 S clips + extra DailyDialog clips + S re-samples."""
import glob, json
from collections import Counter
SV = "/scratch/pa2753/synvid"; A = f"{SV}/a18"
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
W = set(open(f"{SV}/v0/split/W.txt").read().split())
S = jl(f"{SV}/v0/split/roles/v0_S.jsonl")
out = list(jl(f"{SV}/v3/v3_S_gen.jsonl")); src = Counter(s_clips=len(out))
for g in jl(f"{A}/x/rows.jsonl"):
    assert g["valid"]
    out.append(dict(id=g["sid"], take_id=g["sid"], rel_path=g["sid"] + ".npz", domain="gen_dd", raw_sentence=g["transcript"],
                    transcript=g["transcript"], norm=g["transcript"], video_path=g["npz_path"])); src["dd_extra"] += 1
for d in sorted(glob.glob(f"{A}/r*/")):
    it = {i["sid"]: i for i in jl(d + "items.jsonl")}
    for g in jl(d + "rows.jsonl"):
        assert g["valid"]; r = S[it[g["sid"]]["s_index"]]; assert g["transcript"] == r["norm"]
        out.append(dict(id=r["id"] + "__" + g["sid"], take_id=g["sid"], rel_path=g["sid"] + ".npz", domain="gen_resample",
                        raw_sentence=r["raw_sentence"], transcript=r["transcript"], norm=r["norm"], video_path=g["npz_path"]))
        src["resample"] += 1
cov = Counter(w for r in out for w in set(r["norm"].split()) & W)
open(f"{A}/a18_A3x5.jsonl", "w").write("".join(json.dumps(o) + "\n" for o in out))
rep = dict(rows=len(out), by_source=src, min_W_cov=min(cov[w] for w in W), W_below5=sorted(w for w in W if cov[w] < 5))
json.dump(rep, open(f"{A}/manifest_report.json", "w"), indent=1); print(json.dumps(rep))
