"""Experiment R: does the finetuned model's quality depend on WHICH reference clips are used at generation?

    python ref_eval.py plans                  # per-fold, per-reference plans -> cvplan/plan_ref.json (builds run folders)
    python ref_eval.py items --chunks N       # PathBench items for the NEW runs + neural-job list in <ROOT>/exp_R/
    python ref_eval.py table                  # table + R1/R2 verdicts -> <ROOT>/exp_R/table.{md,json}

Same cross-fitted held-out set as experiment K (229 sentences, 5 folds, the fold adapters A/B/C at their chosen
epochs, seeds 1-3), generated with four reference prompts instead of one: ft_stitch5 (the K runs, reused), stitch5
(original 5-clip stitch), lowpitch5 (5 deliberately low-pitched clips) and median1 (one clip). Every source clip of
every reference is excluded from the held-out pool by cv_prep.py (it excludes sources of all refs in refs_ft.json)
and is in every training set, so no reference contains a held-out sentence. Control Z(r) = pretrained model with the
same reference r, same code path, same seeds. Criteria R1/R2 are pre-registered in README.md (Experiment R).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_metrics as M  # noqa: E402
import multi_eval as ME  # noqa: E402

REFS = ["ft_stitch5", "stitch5", "lowpitch5", "median1"]
NEW_REFS = REFS[1:]
LABS = ["A", "B", "C", "Z"]
EPOCH = {"A_r16": "epoch_12", "B_r64": "epoch_08", "C_r16_lr5e-5": "epoch_12"}
SEEDS = [1, 2, 3]
OUT = f"{M.ROOT}/exp_R"
NBOOT = ME.NBOOT


def run_name(lab: str, ref: str, k: int, s: int) -> str:
    tag = "pretrained" if lab == "Z" else ME.FTS[lab]
    return f"FT_{tag}_cv{k}_{ref}_tts_test_s{s}"


def define() -> dict:
    fold_targets, fold_of, train_norms, targets = {}, {}, {}, {}
    for k in range(ME.K_FOLDS):
        sp = ME.cv_splits(k)
        ft = M.split_targets(sp, "tts_test")
        fold_targets[k] = ft
        targets.update(ft)
        fold_of.update({n: k for n in ft})
        train_norms[k] = {sp["clips"][u]["norm"] for u in sp["tts_train"]}
    groups = {(ref, lab): {s: [(run_name(lab, ref, k, s), fold_targets[k]) for k in range(ME.K_FOLDS)] for s in SEEDS}
              for ref in REFS for lab in LABS}
    return {"targets": dict(sorted(targets.items())), "groups": groups, "fold_of": fold_of, "train_norms": train_norms,
            "base": M.load_splits()}


def all_runs(d: dict, refs=REFS) -> list[str]:
    return sorted({r for (ref, _), g in d["groups"].items() if ref in refs for ents in g.values() for r, _ in ents})


def cmd_plans(_a) -> None:
    refs_json = f"{M.ROOT}/refs/refs_ft.json"
    refs = json.load(open(refs_json))
    for r in REFS:
        assert r in (refs.get("refs") or refs), f"{r} missing from {refs_json}"
    os.makedirs(f"{M.ROOT}/cvplan", exist_ok=True)
    units, model = [], None
    for ref in NEW_REFS:
        for k in range(ME.K_FOLDS):
            out = f"{M.ROOT}/cvplan/plan_ref_{ref}_cv{k}.json"
            cmd = [sys.executable, f"{HERE}/build_plan_ft.py", "--out", out, "--split", "tts_test", "--splits-file",
                   f"{M.ROOT}/data/cv/splits_cv{k}.json", "--ref", ref, "--seeds", *map(str, SEEDS)]
            for lab in ("A", "B", "C"):
                run = ME.FTS[lab]
                cmd += ["--adapter", f"{run}_cv{k}={M.ROOT}/runs/{run}_cv{k}/{EPOCH[run]}"]
            cmd += ["--adapter", f"pretrained_cv{k}=none"]
            subprocess.run(cmd, check=True)
            p = json.load(open(out))
            model = model or p["model_path"]
            units += p["units"]
    assert len({u["name"] for u in units}) == len(units), "duplicate unit names"
    json.dump({"model_path": model, "units": units}, open(f"{M.ROOT}/cvplan/plan_ref.json", "w"), indent=1)
    print(len(units), "units,", sum(len(u["prompts"]) for u in units), "clips -> cvplan/plan_ref.json")


def cmd_items(a) -> None:
    d = define()
    os.makedirs(OUT, exist_ok=True)
    items, problems = [], []
    for run in all_runs(d, NEW_REFS):
        tg = next(t for g in d["groups"].values() for ents in g.values() for r, t in ents if r == run)
        c = M.run_completeness(run, tg)
        if not c["ok"]:
            problems.append(f"{run}: {c['n_present']}/{c['n_targets']}, missing audio {len(c['missing_audio'])}")
            continue
        items += M.synth_items(run, run, tg)
    if problems and not a.allow_incomplete:
        sys.exit("incomplete generation:\n  " + "\n  ".join(problems))
    M.write_csv(f"{OUT}/pb_items_all.csv", items, M.ITEM_FIELDS)
    for i in range(a.chunks):
        M.write_csv(f"{OUT}/pb_items_{i:02d}.csv", items[i :: a.chunks], M.ITEM_FIELDS)
    todo = [f"ms_{r}|{r}" for r in all_runs(d, NEW_REFS) if not os.path.isdir(ME.neural_dir(r)) or
            not any(f.startswith("neural.shard") for f in os.listdir(ME.neural_dir(r)))]
    open(f"{OUT}/neural_todo.txt", "w").write("\n".join(todo) + ("\n" if todo else ""))
    print(f"{len(items)} PathBench items in {a.chunks} chunks; {len(todo)} runs need neural embeddings")


def table(_a) -> None:
    d = define()
    pb_dirs = [f"{OUT}/pb_out", f"{M.ROOT}/exp_K/pb_out"]  # K already scored the ft_stitch5 runs and the real clips
    pb = M.load_pb_scored(M.pb_files(*[x for x in pb_dirs if os.path.isdir(x)]))
    runs = all_runs(d)
    dirs = sorted({ME.neural_dir(r) for r in runs})
    for x in dirs + [M.REAL_BANK_DIR]:
        if not os.path.isdir(x):
            sys.exit(f"{x} missing (neural scoring not run for it?)")
    store, bank = M.EmbStore(dirs), M.EmbStore([M.REAL_BANK_DIR])
    targets = d["targets"]
    real = M.score_real(targets, pb, bank, M.references_excluded(d["groups"][("ft_stitch5", "A")][1][0][0]))
    R = {m: real["_sent"][m] for m in ("per", "artp")}

    comb: dict = {}
    for key, g in d["groups"].items():
        comb[key] = {}
        for s, ents in g.items():
            recs = [M.score_run(r, r, tg, pb, store, bank) for r, tg in ents]
            bad = [r["label"] for r in recs if r["n_synth"] < r["n_targets"]]
            if bad:
                print(f"WARNING incomplete: {bad}", flush=True)
            comb[key][s] = ME.combine(recs)

    def sent_mean(key, m: str) -> dict[str, float]:
        recs = [r["_sent"][m] for r in comb[key].values()]
        ks = set.intersection(*[set(x) for x in recs])
        return {k: float(np.mean([x[k] for x in recs])) for k in ks}

    def agg(key) -> dict:
        rs = list(comb[key].values())
        o = {}
        for k in ("per", "artp", "titanet", "wavlm", "dur_median", "dur_within20", "dur_ramble"):
            v = np.array([r[k] for r in rs], float)
            o[k] = float(np.nanmean(v))
            o[k + "_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
        return o

    A = {key: agg(key) for key in comb}
    common = set(R["per"]) & set(R["artp"])
    for key in comb:
        common &= set(sent_mean(key, "per")) & set(sent_mean(key, "artp"))
    keys = sorted(common)
    n_all, n_keys = len(targets), len(keys)
    dper = {key: np.array([sent_mean(key, "per")[k] - R["per"][k] for k in keys]) for key in comb}
    dart = {key: np.array([sent_mean(key, "artp")[k] - R["artp"][k] for k in keys]) for key in comb}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_keys, (NBOOT, n_keys))

    # R1 per (finetuned run, reference): absolute quality against the same-reference zero-shot control
    ver: dict = {}
    for lab in ("A", "B", "C"):
        ver[lab] = {"R1": {}, "R3": {}}
        for ref in REFS:
            x, z, a_, za = dper[(ref, lab)], dper[(ref, "Z")], A[(ref, lab)], A[(ref, "Z")]
            crit = {"PER": abs(float(x.mean())) <= ME.T_PER_GAP, "voice": a_["titanet"] >= ME.T_TITANET,
                    "length": a_["dur_within20"] >= ME.T_WITHIN, "ramble": a_["dur_ramble"] < za["dur_ramble"]}
            ver[lab]["R1"][ref] = {"pass": bool(all(crit.values())), "criteria": {k: bool(v) for k, v in crit.items()},
                                   "bias": float(x.mean()), "titanet": a_["titanet"], "within20": a_["dur_within20"],
                                   "ramble": a_["dur_ramble"], "ramble_Z": za["dur_ramble"]}
            diff = np.abs(x[idx].mean(1)) - np.abs(z[idx].mean(1))
            mae = np.abs(x[idx]).mean(1) - np.abs(z[idx]).mean(1)
            ver[lab]["R3"][ref] = {"d_abs_bias": float(abs(x.mean()) - abs(z.mean())), "ci_abs_bias": [float(v) for v in np.percentile(diff, [2.5, 97.5])],
                                   "d_mae": float(np.abs(x).mean() - np.abs(z).mean()), "ci_mae": [float(v) for v in np.percentile(mae, [2.5, 97.5])]}
        ver[lab]["R1_overall"] = "PASS" if all(v["pass"] for v in ver[lab]["R1"].values()) else "FAIL"
        # R2: spread of mean PER across the references, finetuned minus same-reference zero-shot
        def spread_pt(l):
            m = [dper[(r, l)].mean() for r in REFS]
            return float(max(m) - min(m))

        def spread_boot(l):
            m = np.stack([dper[(r, l)][idx].mean(1) for r in REFS])
            return m.max(0) - m.min(0)

        sx, sz = spread_pt(lab), spread_pt("Z")
        bx, bz = spread_boot(lab), spread_boot("Z")
        lo, hi = (float(v) for v in np.percentile(bx - bz, [2.5, 97.5]))
        ver[lab]["R2"] = {"pass": bool(hi < 0), "range_FT": sx, "range_Z": sz, "d": sx - sz, "ci": [lo, hi]}
        ver[lab]["R2_overall"] = "PASS" if hi < 0 else "FAIL"

    names = {"A": "Finetuned A", "B": "Finetuned B", "C": "Finetuned C", "Z": "Zero-shot, same reference"}
    md = ["# Experiment R: reference sensitivity", "",
          f"{n_all} held-out sentences from the K folds ({n_keys} with PER and ArtP in every row), seeds {SEEDS}. Real PER {real['per']:.3f}, "
          f"ArtP {real['artp']:.3f}, TitaNet ceiling {real['titanet']:.3f}. Criteria R1/R2 are pre-registered (README, Experiment R).", ""]
    for ref in REFS:
        md += [f"## Reference: {ref}", "", "| row | PER | bias | ArtP | TitaNet | within +-20% | ramble |", "|---|---|---|---|---|---|---|"]
        for lab in LABS:
            x = A[(ref, lab)]
            md.append(f"| {names[lab]} | {x['per']:.3f} +- {x['per_sd']:.3f} | {dper[(ref, lab)].mean():+.3f} | {x['artp']:.3f} | {x['titanet']:.3f} | "
                      f"{100 * x['dur_within20']:.0f}% | {100 * x['dur_ramble']:.1f}% |")
        md.append("")
    md += ["## R1: quality holds for each reference (bias <= 0.07, TitaNet >= 0.64, within +-20% >= 55%, ramble < same-reference zero-shot)", ""]
    for lab in ("A", "B", "C"):
        md.append(f"- {names[lab]}: **{ver[lab]['R1_overall']}**  " + "; ".join(
            f"{ref} {'PASS' if v['pass'] else 'FAIL (' + ','.join(k for k, ok in v['criteria'].items() if not ok) + ')'}" for ref, v in ver[lab]["R1"].items()))
    md += ["", "## R2: finetuning reduces the spread of mean PER across references (range FT - range Z, 95% CI must be < 0)", "",
           "| run | range FT | range Z | difference [95% CI] | verdict |", "|---|---|---|---|---|"]
    for lab in ("A", "B", "C"):
        v = ver[lab]["R2"]
        md.append(f"| {names[lab]} | {v['range_FT']:.3f} | {v['range_Z']:.3f} | {v['d']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | **{ver[lab]['R2_overall']}** |")
    md += ["", "## Secondary: per reference, finetuned vs same-reference zero-shot (paired over sentences, 95% CI)", "",
           "| run | reference | |bias| FT - Z | per-sentence MAE FT - Z |", "|---|---|---|---|"]
    for lab in ("A", "B", "C"):
        for ref in REFS:
            v = ver[lab]["R3"][ref]
            md.append(f"| {names[lab]} | {ref} | {v['d_abs_bias']:+.3f} [{v['ci_abs_bias'][0]:+.3f}, {v['ci_abs_bias'][1]:+.3f}] | "
                      f"{v['d_mae']:+.3f} [{v['ci_mae'][0]:+.3f}, {v['ci_mae'][1]:+.3f}] |")
    md += ["", "## Spread across references (max - min of the four reference means)", "", "| row | PER | ArtP | TitaNet | within +-20% |", "|---|---|---|---|---|"]
    for lab in LABS:
        sp = lambda k: max(A[(r, lab)][k] for r in REFS) - min(A[(r, lab)][k] for r in REFS)  # noqa: E731
        md.append(f"| {names[lab]} | {sp('per'):.3f} | {sp('artp'):.3f} | {sp('titanet'):.3f} | {100 * sp('dur_within20'):.0f} pp |")
    open(f"{OUT}/table.md", "w").write("\n".join(md) + "\n")
    json.dump(M.strip_private({"n_sentences": n_all, "n_complete": n_keys, "real": real,
                               "agg": {f"{r}|{l}": v for (r, l), v in A.items()}, "verdicts": ver}), open(f"{OUT}/table.json", "w"), indent=1)
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plans")
    i = sub.add_parser("items")
    i.add_argument("--chunks", type=int, default=12)
    i.add_argument("--allow-incomplete", action="store_true")
    sub.add_parser("table")
    a = ap.parse_args()
    {"plans": cmd_plans, "items": cmd_items, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
