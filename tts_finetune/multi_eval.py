"""Follow-up experiments S (more seeds, tts_test) and K (5-fold cross-fitted larger held-out set); protocol and the
pre-registered criteria E1-E4 are in README.md ("Follow-up experiments").

    python multi_eval.py items --exp S|K --chunks N   # PathBench items + neural-job list in <ROOT>/exp_<EXP>/
    python multi_eval.py table --exp S|K              # table + E1-E4 verdicts -> <ROOT>/exp_<EXP>/table.{md,json}

Run groups (one generated run folder per row of RUNS below; for K one folder per fold):
  A/B/C   the finetuned runs at their chosen epochs (S: the Phase 4 adapters; K: the fold models)
  Z2      pretrained model + ft_stitch5 (same reference as the finetuned runs)
  Z1      zero-shot stitch5, original reference (existing seeds 1-3, plus S: seeds 4-10 generated with gen_ft 'none')
  single-seed zero-shot rows: stitch10, lowpitch5, median1 (existing full-scale runs restricted to the same sentences)
The unit of the paired statistics is the SENTENCE: d_i = (mean over seeds of synthetic per-sentence value) minus the
real per-sentence value (mean over the real repeats).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_metrics as M  # noqa: E402

K_FOLDS = 5
NBOOT = 10000
T_PER_GAP, T_TITANET, T_WITHIN, T_ARTP = 0.07, 0.64, 0.55, 0.04
FTS = {"A": "A_r16", "B": "B_r64", "C": "C_r16_lr5e-5"}   # label -> run (adapter tag stem)
SINGLES = {"stitch10": "R1b_stitch10_s1", "lowpitch5": "LP_low5_s1", "median1": "R2_median1_s1"}
OUT_V2 = M.OUT_V2


def exp_dir(exp: str) -> str:
    return f"{M.ROOT}/exp_{exp}"


def cv_splits(k: int) -> dict:
    return json.load(open(f"{M.ROOT}/data/cv/splits_cv{k}.json"))


# ----------------------------------------------------------------------------- experiment definitions
def define(exp: str) -> dict:
    """targets (sentence set) and the run folders of every group.
    groups[label] = {"seeds": {seed: [(run_name, targets_of_that_run)]}}   (K: one entry per fold)"""
    base = M.load_splits()
    groups: dict[str, dict] = {}
    if exp == "S":
        targets = M.split_targets(base, "tts_test")
        seeds = list(range(1, 11))
        for lab, run in FTS.items():
            groups[lab] = {s: [(f"FT_{run}_best_ft_stitch5_tts_test_s{s}", targets)] for s in seeds}
        groups["Z2"] = {s: [(f"FT_pretrained_ft_stitch5_tts_test_s{s}", targets)] for s in seeds}
        groups["Z1"] = {s: [(f"R1_stitch5_s{s}" if s <= 3 else f"FT_zs_stitch5_stitch5_tts_test_s{s}", targets)] for s in seeds}
        train_norms = {"all": {base["clips"][u]["norm"] for u in base["tts_train"]}}
        fold_of = {n: "all" for n in targets}
    elif exp == "K":
        targets, fold_targets, fold_of, train_norms = {}, {}, {}, {}
        for k in range(K_FOLDS):
            sp = cv_splits(k)
            ft = M.split_targets(sp, "tts_test")
            fold_targets[k] = ft
            targets.update(ft)
            fold_of.update({n: k for n in ft})
            train_norms[k] = {sp["clips"][u]["norm"] for u in sp["tts_train"]}
        assert sum(len(t) for t in fold_targets.values()) == len(targets), "a sentence is held out in two folds"
        targets = dict(sorted(targets.items()))
        seeds = [1, 2, 3]
        for lab, run in FTS.items():
            groups[lab] = {s: [(f"FT_{run}_cv{k}_ft_stitch5_tts_test_s{s}", fold_targets[k]) for k in range(K_FOLDS)] for s in seeds}
        groups["Z2"] = {s: [(f"FT_pretrained_cv{k}_ft_stitch5_tts_test_s{s}", fold_targets[k]) for k in range(K_FOLDS)] for s in seeds}
        groups["Z1"] = {s: [(f"R1_stitch5_s{s}", targets)] for s in seeds}
    else:
        raise SystemExit("--exp must be S or K")
    for lab, run in SINGLES.items():
        groups[lab] = {1: [(run, targets)]}
    return {"exp": exp, "targets": targets, "groups": groups, "fold_of": fold_of, "train_norms": train_norms, "base": base}


def neural_dir(run: str) -> str:
    """v2-style neural shard dir of a run: the existing Phase 4 / baseline dirs, else out_v2/ms_<run>."""
    cands = [f"{OUT_V2}/ms_{run}", f"{OUT_V2}/gen_{run}"]
    m = re.match(r"FT_(A_r16|B_r64|C_r16_lr5e-5)_best_ft_stitch5_tts_test_s(\d+)$", run)
    if m:
        cands.insert(0, f"{OUT_V2}/ft_{m.group(1)}_best_s{m.group(2)}")
    return next((c for c in cands if os.path.isdir(c)), cands[0])


def all_runs(d: dict) -> list[str]:
    return sorted({r for g in d["groups"].values() for ents in g.values() for r, _ in ents})


# ----------------------------------------------------------------------------- items stage
def cmd_items(a) -> None:
    d = define(a.exp)
    out = exp_dir(a.exp)
    os.makedirs(out, exist_ok=True)
    items = M.real_items(d["targets"], d["base"])
    problems = []
    for g in d["groups"].values():
        for ents in g.values():
            for run, tg in ents:
                c = M.run_completeness(run, tg)
                if not c["ok"]:
                    problems.append(f"{run}: {c['n_present']}/{c['n_targets']} sentences, missing audio {len(c['missing_audio'])}")
                    continue
                items += M.synth_items(run, run, tg)
    if problems and not a.allow_incomplete:
        sys.exit("incomplete generation:\n  " + "\n  ".join(problems))
    # dedupe baselines that appear once per condition (same audio_path+label)
    seen, uniq = set(), []
    for r in items:
        k = (r["severity"], r["audio_path"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    M.write_csv(f"{out}/pb_items_all.csv", uniq, M.ITEM_FIELDS)
    for i in range(a.chunks):
        M.write_csv(f"{out}/pb_items_{i:02d}.csv", uniq[i :: a.chunks], M.ITEM_FIELDS)
    todo = [f"ms_{r}|{r}" for r in all_runs(d) if not os.path.isdir(neural_dir(r)) or
            not any(f.startswith("neural.shard") for f in os.listdir(neural_dir(r)))]
    open(f"{out}/neural_todo.txt", "w").write("\n".join(todo) + ("\n" if todo else ""))
    print(f"{len(uniq)} PathBench items in {a.chunks} chunks; {len(todo)} runs need neural embeddings -> {out}/neural_todo.txt")


# ----------------------------------------------------------------------------- stats
def wed(a: list[str], b: list[str]) -> int:
    dd = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, dd[0] = dd[0], i
        for j, y in enumerate(b, 1):
            p, dd[j] = dd[j], min(dd[j] + 1, dd[j - 1] + 1, p + (x != y))
    return dd[len(b)]


def combine(recs: list[dict]) -> dict:
    """Merge score_run records of the same (group, seed) over folds: sentence dicts are unioned, clip-weighted voice."""
    sent: dict[str, dict] = {}
    for r in recs:
        for m, dct in r["_sent"].items():
            sent.setdefault(m, {}).update(dct)
    out = {"per": M.mean_or_nan(sent["per"].values()), "artp": M.mean_or_nan(sent["artp"].values()),
           "n_synth": sum(r["n_synth"] for r in recs), "n_targets": sum(r["n_targets"] for r in recs), "_sent": sent}
    ds = M.duration_stats(list(sent["dur_ratio"].values()))
    out.update({f"dur_{k}": v for k, v in ds.items()})
    for m in M.SV_MODELS:
        n = sum(r[f"{m}_n"] for r in recs)
        out[m] = float(sum(r[m] * r[f"{m}_n"] for r in recs if r[f"{m}_n"]) / n) if n else float("nan")
    return out


def boot_ci(pairs: np.ndarray, fn, rng) -> tuple[float, float]:
    n = len(pairs)
    idx = rng.integers(0, n, (NBOOT, n))
    v = fn(pairs[idx])  # (NBOOT, n, 2)
    return tuple(float(x) for x in np.percentile(v, [2.5, 97.5]))


def fb_(p: np.ndarray) -> np.ndarray:
    return np.abs(p[:, :, 0].mean(1)) - np.abs(p[:, :, 1].mean(1))


def table(d: dict, a) -> None:
    out = exp_dir(a.exp)
    pb = M.load_pb_scored(M.pb_files(*[f"{out}/pb_out"] + ([f"{out}/pb_out_extra"] if os.path.isdir(f"{out}/pb_out_extra") else [])))
    runs = all_runs(d)
    dirs = sorted({neural_dir(r) for r in runs})
    for x in dirs + [M.REAL_BANK_DIR]:
        if not os.path.isdir(x):
            sys.exit(f"{x} missing (neural scoring not run for it?)")
    store, bank = M.EmbStore(dirs), M.EmbStore([M.REAL_BANK_DIR])
    targets = d["targets"]
    first = d["groups"]["A"][1][0][0]
    real = M.score_real(targets, pb, bank, M.references_excluded(first))
    R = {m: real["_sent"][m] for m in ("per", "artp")}

    comb: dict[str, dict[int, dict]] = {}
    for lab, g in d["groups"].items():
        comb[lab] = {}
        for s, ents in g.items():
            recs = [M.score_run(r, r, tg, pb, store, bank) for r, tg in ents]
            bad = [r["label"] for r in recs if r["n_synth"] < r["n_targets"]]
            if bad:
                print(f"WARNING incomplete: {bad}", flush=True)
            comb[lab][s] = combine(recs)

    names = {"A": "A_r16 (e12)", "B": "B_r64 (e8)", "C": "C_r16_lr5e-5 (e12)", "Z2": "zero-shot, pretrained + ft_stitch5 (Z2)",
             "Z1": "zero-shot stitch5, original ref (Z1)", "stitch10": "zero-shot stitch10", "lowpitch5": "zero-shot lowpitch5",
             "median1": "zero-shot median1"}

    def sent_mean(lab: str, m: str) -> dict[str, float]:
        recs = [r["_sent"][m] for r in comb[lab].values()]
        keys = set.intersection(*[set(x) for x in recs])
        return {k: float(np.mean([x[k] for x in recs])) for k in keys}

    def agg(lab: str) -> dict:
        rs = list(comb[lab].values())
        f = lambda k: [r[k] for r in rs]  # noqa: E731
        o = {"n_seeds": len(rs)}
        for k in ("per", "artp", "titanet", "wavlm", "dur_median", "dur_within20", "dur_ramble"):
            v = np.array(f(k), float)
            o[k] = float(np.nanmean(v))
            o[k + "_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
        return o

    A = {lab: agg(lab) for lab in comb}
    rng = np.random.default_rng(0)
    common = set(R["per"]) & set(R["artp"])  # a sentence needs PER and ArtP in every row (ArtP can be missing for very short clips)
    for lab in comb:
        common &= set(sent_mean(lab, "per")) & set(sent_mean(lab, "artp"))
    keys = sorted(common)
    if len(keys) < len(targets):
        print(f"NOTE: {len(targets) - len(keys)} of {len(targets)} sentences lack PER or ArtP in some row and are left out of the paired statistics", flush=True)
    dper = {lab: np.array([sent_mean(lab, "per")[k] - R["per"][k] for k in keys]) for lab in comb}
    dart = {lab: np.array([sent_mean(lab, "artp")[k] - R["artp"][k] for k in keys]) for lab in comb if keys}
    near = np.array([min(wed(k.split(), t.split()) for t in d["train_norms"][d["fold_of"][k]]) <= 1 for k in keys])

    zero_rows = ["Z1", "Z2", "stitch10", "lowpitch5", "median1"]
    ver: dict[str, dict] = {}
    for lab in ("A", "B", "C"):
        dx, a_ = dper[lab], A[lab]
        gap = abs(float(dx.mean()))
        pair1 = np.stack([dx, dper["Z1"]], 1)
        fb = lambda p: np.abs(p[:, :, 0].mean(1)) - np.abs(p[:, :, 1].mean(1))  # noqa: E731
        fm = lambda p: np.abs(p[:, :, 0]).mean(1) - np.abs(p[:, :, 1]).mean(1)  # noqa: E731
        lo1, hi1 = boot_ci(pair1, fb, rng)
        pair2 = np.stack([dx, dper["Z2"]], 1)
        lo2, hi2 = boot_ci(pair2, fb, rng)
        ml1 = boot_ci(pair1, fm, rng)
        ml2 = boot_ci(pair2, fm, rng)
        ramble_base = {z: A[z]["dur_ramble"] for z in zero_rows}
        crit = {
            "E1": (gap <= T_PER_GAP, f"|mean PER - real| = {gap:.3f} <= {T_PER_GAP} (bias {dx.mean():+.3f})"),
            "E2": (hi1 < 0, f"|bias| FT - Z1 = {abs(dx.mean()) - abs(dper['Z1'].mean()):+.3f}, 95% CI [{lo1:+.3f}, {hi1:+.3f}] must be < 0"),
            "E3a": (a_["titanet"] >= T_TITANET, f"TitaNet {a_['titanet']:.3f} >= {T_TITANET}"),
            "E3b": (a_["dur_within20"] >= T_WITHIN, f"within +-20% {100 * a_['dur_within20']:.0f}% >= {100 * T_WITHIN:.0f}%"),
            "E3c": (all(a_["dur_ramble"] < v for v in ramble_base.values()),
                    f"ramble {100 * a_['dur_ramble']:.1f}% < each zero-shot ({', '.join(f'{z} {100 * v:.1f}%' for z, v in ramble_base.items())})"),
            "E4": (abs(float(dart[lab].mean())) <= T_ARTP, f"|mean ArtP - real| = {abs(float(dart[lab].mean())):.3f} <= {T_ARTP}"),
        }
        ver[lab] = {"overall": "PASS" if all(v[0] for v in crit.values()) else "FAIL",
                    "criteria": {k: {"pass": bool(v[0]), "text": v[1]} for k, v in crit.items()},
                    "secondary_vs_Z2": {"d_abs_bias": float(abs(dx.mean()) - abs(dper["Z2"].mean())), "ci": [lo2, hi2]},
                    "mae_vs_Z1": {"d": float(np.abs(dx).mean() - np.abs(dper["Z1"]).mean()), "ci": list(ml1)},
                    "mae_vs_Z2": {"d": float(np.abs(dx).mean() - np.abs(dper["Z2"]).mean()), "ci": list(ml2)}}

    n_s, n_c = len(targets), real["n_clips"]
    md = [f"# Experiment {a.exp}: {'more seeds on tts_test' if a.exp == 'S' else '5-fold cross-fitted held-out set'}", "",
          f"{n_s} sentences ({len(keys)} with complete data in every row), {n_c} real clips; real PER {real['per']:.3f}, ArtP {real['artp']:.3f}, "
          f"TitaNet ceiling {real['titanet']:.3f}. Cells: mean over seeds +- sd over seeds (seeds: "
          f"{sorted(d['groups']['A'])}). Generated by `multi_eval.py table`; criteria are the pre-registered E1-E4 (README).", "",
          "| row | seeds | PER | ArtP | TitaNet | WavLM | dur ratio (median) | within +-20% | ramble (>2x) |", "|---|---|---|---|---|---|---|---|---|"]
    pm = lambda x, s, p=3: f"{x:.{p}f}" + (f" +- {s:.{p}f}" if np.isfinite(s) else "")  # noqa: E731
    pc = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    for lab in ["A", "B", "C", "Z2", "Z1", "stitch10", "lowpitch5", "median1"]:
        x = A[lab]
        md.append(f"| {names[lab]} | {x['n_seeds']} | {pm(x['per'], x['per_sd'])} | {pm(x['artp'], x['artp_sd'])} | {pm(x['titanet'], x['titanet_sd'])} | "
                  f"{pm(x['wavlm'], x['wavlm_sd'])} | {x['dur_median']:.2f} | {pc(x['dur_within20'])} | {100 * x['dur_ramble']:.1f}% |")
    md += ["", "## Pre-registered criteria", ""]
    for lab, v in ver.items():
        md += [f"### {names[lab]}: **{v['overall']}**", ""]
        for k, c in v["criteria"].items():
            md.append(f"- **{'PASS' if c['pass'] else 'FAIL'}** {k}: {c['text']}")
        s2 = v["secondary_vs_Z2"]
        md += [f"- (secondary, not in verdict) |bias| FT - Z2 = {s2['d_abs_bias']:+.3f}, 95% CI [{s2['ci'][0]:+.3f}, {s2['ci'][1]:+.3f}]", ""]
    md += ["## Per-sentence error (paired over sentences, bootstrap 95% CI)", "",
           "| row | bias PER | MAE PER | MAE - MAE(Z1) [95% CI] | MAE - MAE(Z2) [95% CI] | bias ArtP | MAE ArtP |", "|---|---|---|---|---|---|---|"]
    for lab in ["A", "B", "C", "Z2", "Z1", "stitch10", "lowpitch5", "median1"]:
        v = ver.get(lab)
        c1 = f"{v['mae_vs_Z1']['d']:+.3f} [{v['mae_vs_Z1']['ci'][0]:+.3f}, {v['mae_vs_Z1']['ci'][1]:+.3f}]" if v else "-"
        c2 = f"{v['mae_vs_Z2']['d']:+.3f} [{v['mae_vs_Z2']['ci'][0]:+.3f}, {v['mae_vs_Z2']['ci'][1]:+.3f}]" if v else "-"
        md.append(f"| {names[lab]} | {dper[lab].mean():+.3f} | {np.abs(dper[lab]).mean():.3f} | {c1} | {c2} | {dart[lab].mean():+.3f} | {np.abs(dart[lab]).mean():.3f} |")
    md += ["", f"## Near-duplicate vs far sentences (word edit distance <= 1 to a training sentence of that run): near n={int(near.sum())}, far n={int((~near).sum())}", "",
           "| row | near: bias / MAE | far: bias / MAE |", "|---|---|---|"]
    for lab in ["A", "B", "C", "Z2", "Z1"]:
        f = lambda m: f"{dper[lab][m].mean():+.3f} / {np.abs(dper[lab][m]).mean():.3f}" if m.any() else "-"  # noqa: E731
        md.append(f"| {names[lab]} | {f(near)} | {f(~near)} |")
    if a.exp == "K":
        base_test = {d["base"]["clips"][u]["norm"] for u in d["base"]["tts_test"]}
        nov = np.array([k not in base_test for k in keys])
        md += ["", f"## Sentences never in the primary tts_test (n={int(nov.sum())} of {len(keys)}); criteria E1, E2, E4 (PER/ArtP-based) on this subset", ""]
        sub = {}
        for lab in ("A", "B", "C"):
            dx, dz = dper[lab][nov], dper["Z1"][nov]
            lo, hi = boot_ci(np.stack([dx, dz], 1), fb_, rng)
            e1, e2, e4 = abs(dx.mean()) <= T_PER_GAP, hi < 0, abs(dart[lab][nov].mean()) <= T_ARTP
            sub[lab] = {"E1": bool(e1), "E2": bool(e2), "E4": bool(e4), "bias": float(dx.mean()), "ci_vs_Z1": [lo, hi]}
            md.append(f"- {names[lab]}: E1 {'PASS' if e1 else 'FAIL'} (|bias| {abs(dx.mean()):.3f}); E2 {'PASS' if e2 else 'FAIL'} "
                      f"(|bias| FT - Z1 {abs(dx.mean()) - abs(dz.mean()):+.3f}, CI [{lo:+.3f}, {hi:+.3f}]); E4 {'PASS' if e4 else 'FAIL'} "
                      f"(|ArtP bias| {abs(dart[lab][nov].mean()):.3f}); Z1 |bias| {abs(dz.mean()):.3f}")
        ver["_novel_subset"] = sub
        fold = np.array([d["fold_of"][k] for k in keys])
        md += ["", "## Per-fold PER bias (synthetic - real), heterogeneity check", "", "| row | " + " | ".join(f"fold {k}" for k in range(K_FOLDS)) + " |", "|---|" + "---|" * K_FOLDS]
        for lab in ["A", "B", "C", "Z2", "Z1"]:
            md.append(f"| {names[lab]} | " + " | ".join(f"{dper[lab][fold == k].mean():+.3f}" for k in range(K_FOLDS)) + " |")
    open(f"{out}/table.md", "w").write("\n".join(md) + "\n")
    json.dump(M.strip_private({"exp": a.exp, "n_sentences": n_s, "n_complete": len(keys), "real": real, "agg": A, "verdicts": ver,
                               "thresholds": {"per_gap": T_PER_GAP, "titanet": T_TITANET, "within20": T_WITHIN, "artp_gap": T_ARTP},
                               "bias_per": {k: float(v.mean()) for k, v in dper.items()},
                               "mae_per": {k: float(np.abs(v).mean()) for k, v in dper.items()}}),
              open(f"{out}/table.json", "w"), indent=1)
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("items")
    i.add_argument("--exp", required=True, choices=["S", "K"])
    i.add_argument("--chunks", type=int, default=4)
    i.add_argument("--allow-incomplete", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("--exp", required=True, choices=["S", "K"])
    a = ap.parse_args()
    if a.cmd == "items":
        cmd_items(a)
    else:
        table(define(a.exp), a)


if __name__ == "__main__":
    main()
