"""A14 V0 scorer: errors on held-out-word (W) tokens in T, A1 vs A0, paired clip bootstrap over seeds.

Usage: python v0_score.py [--alpha 0.75] [--seeds 1 2 3]   (run on torch; reads /scratch/pa2753/synvid/v0/runs)
A reference token counts as an error when the Levenshtein alignment substitutes or deletes it.
"""
import argparse, json, random
from pathlib import Path

V = Path("/scratch/pa2753/synvid/v0")


def ref_errors(ref, hyp):
    """Per reference position: True if substituted or deleted in a minimum-edit alignment."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), D[i - 1][j] + 1, D[i][j - 1] + 1)
    err, i, j = [False] * n, n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            err[i - 1] = ref[i - 1] != hyp[j - 1]; i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            err[i - 1] = True; i -= 1
        else:
            j -= 1
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.75")
    ap.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    ap.add_argument("--draws", type=int, default=4000)
    a = ap.parse_args()
    W = set(V.joinpath("split/W.txt").read_text().split())
    a0_vocab = {w for l in V.joinpath("split/roles/v0_A0_train.jsonl").read_text().splitlines() for w in json.loads(l)["norm"].split()}
    per = {}  # (arm, seed) -> rel_path -> (w_err, w_tok, all_err, all_tok, unseen_err, unseen_tok)
    for arm in ("A0", "A1"):
        for s in a.seeds:
            f = V / f"runs/{arm}/seed{s}/a{a.alpha}/evaluation/v0_T.predictions.json"
            d = {}
            for r in json.loads(f.read_text()):
                ref, hyp = r["normalized_ground_truth"].split(), r["normalized_prediction"].split()
                e = ref_errors(ref, hyp)
                isw = [w in W for w in ref]; isu = [w not in W and w not in a0_vocab for w in ref]
                d[r["rel_path"]] = (sum(x and y for x, y in zip(e, isw)), sum(isw), r["edit_distance"], len(ref),
                                    sum(x and y for x, y in zip(e, isu)), sum(isu))
            per[arm, s] = d
    clips = sorted(per["A0", a.seeds[0]])
    assert all(sorted(v) == clips for v in per.values()) and len(clips) == 100
    tot = lambda arm, k: sum(per[arm, s][c][k] for s in a.seeds for c in clips)
    diff = [sum(per["A1", s][c][0] - per["A0", s][c][0] for s in a.seeds) for c in clips]
    rng = random.Random(20261115)
    boot = sorted(sum(diff[rng.randrange(100)] for _ in clips) for _ in range(a.draws))
    lo, hi = boot[int(0.025 * a.draws)], boot[int(0.975 * a.draws) - 1]
    rep = dict(alpha=a.alpha, seeds=a.seeds, W_tokens_per_seed=tot("A0", 1) // len(a.seeds),
               W_errors={arm: tot(arm, 0) for arm in ("A0", "A1")},
               W_errors_by_seed={f"{arm}/{s}": sum(per[arm, s][c][0] for c in clips) for arm in ("A0", "A1") for s in a.seeds},
               diff_A1_minus_A0=sum(diff), ci95=[lo, hi], PASS=hi < 0,
               all_token_WER={arm: round(100 * tot(arm, 2) / tot(arm, 3), 2) for arm in ("A0", "A1")},
               other_unseen={arm: f"{tot(arm, 4)}/{tot(arm, 5)}" for arm in ("A0", "A1")})
    print(json.dumps(rep, indent=1))
    (V / f"score_a{a.alpha}.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
