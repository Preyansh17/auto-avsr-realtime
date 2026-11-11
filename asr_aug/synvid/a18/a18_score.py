"""A18 scorer: W-token errors on T for A0, A1, A3x1, A3x5 (seeds 4-6); paired clip bootstraps; frozen A18 reading."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, "/scratch/pa2753/synvid/v0/code")
from v0_score import ref_errors
V = Path("/scratch/pa2753/synvid/v0")
ap = argparse.ArgumentParser(); ap.add_argument("--alpha", default="0.75"); ap.add_argument("--seeds", nargs="+", default=["4", "5", "6"])
ap.add_argument("--draws", type=int, default=4000); a = ap.parse_args()
W = set(V.joinpath("split/W.txt").read_text().split())
SEEN = {w for l in V.joinpath("split/roles/v0_A0_train.jsonl").read_text().splitlines() if l.strip() for w in json.loads(l)["norm"].split()}
per = {}
ARMS = ("A0", "A1", "A3x1", "A3x5")
for arm in ARMS:
    for s in a.seeds:
        d = {}
        for r in json.loads((V / f"runs/{arm}/seed{s}/a{a.alpha}/evaluation/v0_T.predictions.json").read_text()):
            ref, hyp = r["normalized_ground_truth"].split(), r["normalized_prediction"].split()
            e = ref_errors(ref, hyp); isw = [w in W for w in ref]
            ou = [w not in W and w not in SEEN for w in ref]
            d[r["rel_path"]] = (sum(x and y for x, y in zip(e, isw)), r["edit_distance"], len(ref), sum(x and y for x, y in zip(e, ou)), sum(ou))
        per[arm, s] = d
clips = sorted(per["A0", a.seeds[0]]); assert all(sorted(v) == clips for v in per.values())
tot = lambda arm, k: sum(per[arm, s][c][k] for s in a.seeds for c in clips)
def boot(x, y):
    diff = [sum(per[x, s][c][0] - per[y, s][c][0] for s in a.seeds) for c in clips]
    rng = random.Random(20261115); b = sorted(sum(diff[rng.randrange(len(clips))] for _ in clips) for _ in range(a.draws))
    return sum(diff), [b[int(0.025 * a.draws)], b[int(0.975 * a.draws) - 1]]
E = {arm: tot(arm, 0) for arm in ARMS}
B = {f"{x}-{y}": boot(x, y) for x, y in (("A3x5", "A0"), ("A3x1", "A0"), ("A3x5", "A3x1"), ("A3x5", "A1"), ("A1", "A0"))}
G = E["A0"] - E["A1"]
helps = B["A3x5-A0"][1][1] < 0
if helps and E["A3x5"] <= E["A3x1"]: dec = "track_alive"
elif not B["A3x5-A0"][1][1] < 0 and not E["A3x5"] < E["A3x1"]: dec = "close"
else: dec = "report_and_ask"
rep = dict(alpha=a.alpha, seeds=a.seeds, W_errors=E, W_errors_by_seed={f"{k}/{s}": sum(per[k, s][c][0] for c in clips) for k in ARMS for s in a.seeds},
           diffs={k: dict(diff=v[0], ci95=v[1]) for k, v in B.items()},
           recovery={k: ((E["A0"] - E[k]) / G if G else None) for k in ("A3x1", "A3x5")}, decision=dec,
           other_unseen_errors={k: [tot(k, 3), tot(k, 4)] for k in ARMS},
           all_token_WER={k: round(100 * tot(k, 1) / tot(k, 2), 2) for k in ARMS})
print(json.dumps(rep, indent=1)); (V.parent / "a18" / f"score_a{a.alpha}.json").write_text(json.dumps(rep, indent=1) + "\n")
