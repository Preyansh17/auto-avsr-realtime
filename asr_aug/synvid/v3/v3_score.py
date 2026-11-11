"""A17 V3 scorer: W-token errors on T for A0, A1 (V0), A2 (V1 TTS audio) and A3 (generated video of S); A3 - A0 paired clip bootstrap."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, "/scratch/pa2753/synvid/v0/code")
from v0_score import ref_errors
V = Path("/scratch/pa2753/synvid/v0")
ap = argparse.ArgumentParser(); ap.add_argument("--alpha", default="0.75"); ap.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
ap.add_argument("--draws", type=int, default=4000); a = ap.parse_args()
W = set(V.joinpath("split/W.txt").read_text().split())
per = {}
for arm in ("A0", "A1", "A2", "A3"):
    for s in a.seeds:
        d = {}
        for r in json.loads((V / f"runs/{arm}/seed{s}/a{a.alpha}/evaluation/v0_T.predictions.json").read_text()):
            ref, hyp = r["normalized_ground_truth"].split(), r["normalized_prediction"].split()
            e = ref_errors(ref, hyp); isw = [w in W for w in ref]
            d[r["rel_path"]] = (sum(x and y for x, y in zip(e, isw)), r["edit_distance"], len(ref))
        per[arm, s] = d
clips = sorted(per["A0", a.seeds[0]]); assert all(sorted(v) == clips for v in per.values())
tot = lambda arm, k: sum(per[arm, s][c][k] for s in a.seeds for c in clips)
def boot(x, y):
    diff = [sum(per[x, s][c][0] - per[y, s][c][0] for s in a.seeds) for c in clips]
    rng = random.Random(20261115); b = sorted(sum(diff[rng.randrange(len(clips))] for _ in clips) for _ in range(a.draws))
    return sum(diff), [b[int(0.025 * a.draws)], b[int(0.975 * a.draws) - 1]]
ARMS = ("A0", "A1", "A2", "A3")
E = {arm: tot(arm, 0) for arm in ARMS}
d30, ci30 = boot("A3", "A0"); d31, ci31 = boot("A3", "A1"); d32, ci32 = boot("A3", "A2")
G = E["A0"] - E["A1"]; r = (E["A0"] - E["A3"]) / G if G else None
ok = ci30[1] < 0 and r is not None and r >= 0.5 and E["A3"] < E["A2"]
rep = dict(alpha=a.alpha, seeds=a.seeds, W_errors=E, W_errors_by_seed={f"{k}/{s}": sum(per[k, s][c][0] for c in clips) for k in ARMS for s in a.seeds},
           A3_minus_A0=d30, ci95_A3_minus_A0=ci30, A3_minus_A1=d31, ci95_A3_minus_A1=ci31, A3_minus_A2=d32, ci95_A3_minus_A2=ci32,
           recovery_r=r, decision="PASS" if ok else "FAIL",
           all_token_WER={k: round(100 * tot(k, 1) / tot(k, 2), 2) for k in ARMS})
print(json.dumps(rep, indent=1)); (V.parent / "v3" / f"score_a{a.alpha}.json").write_text(json.dumps(rep, indent=1) + "\n")
