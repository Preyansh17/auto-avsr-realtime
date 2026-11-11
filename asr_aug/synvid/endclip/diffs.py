"""List in-home clips whose arm b/c prediction differs from arm a, with per-clip error and last-word status."""
import json, sys
sys.path.insert(0, "/scratch/pa2753/synvid/endclip")
from endlib import clip_stats
E = "/scratch/pa2753/synvid/endclip"
L = {a: {x["rel_path"]: x for x in json.load(open(f"{E}/dec/{a}/evaluation/dc_home_all.predictions.json"))} for a in "abc"}
for k in sorted(L["a"]):
    if L["b"][k]["prediction"] != L["a"][k]["prediction"] or L["c"][k]["prediction"] != L["a"][k]["prediction"]:
        g = L["a"][k]["ground_truth"]; s = {a: clip_stats(g, L[a][k]["prediction"]) for a in "abc"}
        print(f"REF {g}")
        for a in "abc": print(f"  {a} err={s[a]['err']} lastwrong={s[a]['last_wrong']} | {L[a][k]['prediction']}")
