"""Phase 4: final evaluation of the selected checkpoints on tts_test, 3 generation seeds, vs the zero-shot baselines.

    python final_eval.py plan  --runs A_r16 [B_r64 ...] [--epoch RUN=N]   # generation plan (selected epoch of each run)
    python final_eval.py items [--chunks N]                               # PathBench items + v2 submission wrapper
    python final_eval.py table                                            # final table + PRE-REGISTERED success rule

Flow (accounts / partitions are chosen by the orchestrator and passed on the sbatch command line):

  plan    reads runs/<run>/select.json (written by select_epoch.py select), builds tts_test x seeds {1,2,3} for adapter
          tag <run>_best via build_plan_ft.py -> gen_runs/FT_<run>_best_ft_stitch5_tts_test_s<seed>/ and
          final/plan_test.json + final/final_plan.json (what the later steps read). final_gen.sbatch runs it.
  items   writes final/pb_items_NN.csv (real tts_test clips + every FT run/seed + the zero-shot baselines
          R1_stitch5_s1..s3, R1b_stitch10_s1, LP_low5_s1, R2_median1_s1, all restricted to the tts_test sentences and
          scored against the same text) for final_score.sbatch (one array task per chunk), and final/submit_v2_ft.sh,
          a copy of tts_eval/v2/submit_all.sh whose TABLE holds only the FT runs.
  v2      the prosody + neural (TitaNet, WavLM) evaluation is the EXISTING tts_eval/v2 pipeline. submit_all.sh only
          knows the runs in its built-in TABLE (name|run dir|prosody shards|neural shards|pairs json|runs root), so a
          new run needs a row there. The shared file is not edited: `items` emits final/submit_v2_ft.sh, the same
          script with the TABLE swapped for the FT rows, e.g.
              ft_A_r16_best_s1|FT_A_r16_best_ft_stitch5_tts_test_s1|1|1||/scratch/pa2753/gen_runs
          (pairs json left empty: with manifests every real and synthetic clip is measured and --pairs only adds
          per-sentence ratios, which this script computes itself). Results land in tts_eval/out_v2/<name>/. The v2
          sbatch files carry `#SBATCH --account=torch_pr_39_general`; to spread the jobs over accounts set
          SBATCH_ACCOUNT / SBATCH_PARTITION in the environment of each invocation (sbatch reads them and they take
          precedence over the #SBATCH lines in the script, per the sbatch man page; command-line flags beat both),
          one invocation per run name: `SBATCH_ACCOUNT=<acct> bash final/submit_v2_ft.sh ft_A_r16_best_s1`.
  table   reads the scored csvs only: final/pb_out*/ (PathBench), tts_eval/out_v2/<name>/neural.shard* (embeddings) and
          the runs' metadata csvs (durations). Nothing is typed by hand. Writes final/final_table.{md,csv} and
          final/success_rule.json, and prints PASS/FAIL per criterion with the numbers.

Pre-registered success rule (plan section "Phase 4"), evaluated per FT run on the tts_test sentences, 3 seeds:
  C1  PER gap: mean over seeds of |PER_synth - PER_real| <= 0.07, AND the seed-to-seed spread of PER (max - min over
      the seeds) is smaller than the gap between the FT run's mean PER and the PER of the best zero-shot baseline
      (the baseline condition whose PER is closest to real; stitch5 counts as its 3-seed mean).
  C2  voice: mean TitaNet >= 0.64.
  C3  length: mean share of clips within +-20% of real length >= 55%, AND mean ramble rate (> 2x real) strictly below
      that of EVERY zero-shot baseline row (each of the six listed runs, stitch5 seeds separately).
  C4  ArtP: |mean ArtP_synth - ArtP_real| <= 0.04.
The wording leaves some readings open; the ones taken here are stated next to each criterion in the printed output
(spread = range not sd; gap = FT-vs-best-baseline PER distance; ramble strictly below the minimum). Both alternatives
are printed alongside so a borderline verdict is visible.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_metrics as M  # noqa: E402

SPLIT = "tts_test"
SEEDS = [1, 2, 3]
DEFAULT_REF = "ft_stitch5"
FINAL = os.environ.get("HIGGS_FT_FINAL", f"{M.ROOT}/final")
BUILD_PLAN = os.environ.get("BUILD_PLAN_FT", os.path.join(HERE, "build_plan_ft.py"))
SUBMIT_ALL = os.environ.get("SUBMIT_ALL", f"{M.TTS_EVAL}/v2/submit_all.sh")
BASELINES = ["R1_stitch5_s1", "R1_stitch5_s2", "R1_stitch5_s3", "R1b_stitch10_s1", "LP_low5_s1", "R2_median1_s1"]
CONDITIONS = {"stitch5": ["R1_stitch5_s1", "R1_stitch5_s2", "R1_stitch5_s3"], "stitch10": ["R1b_stitch10_s1"],
              "lowpitch5": ["LP_low5_s1"], "median1": ["R2_median1_s1"]}
T_PER_GAP, T_TITANET, T_WITHIN, T_ARTP = 0.07, 0.64, 0.55, 0.04


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def load_plan() -> dict:
    p = f"{FINAL}/final_plan.json"
    if not os.path.exists(p):
        die(f"{p} missing; run `final_eval.py plan` first")
    return json.load(open(p))


def v2_name(tag: str, seed: int) -> str:
    return f"ft_{tag}_s{seed}"


# ============================================================================= plan
def cmd_plan(a) -> None:
    overrides = {}
    for o in a.epoch or []:
        k, v = o.split("=", 1)
        overrides[k] = int(v)
    splits = M.load_splits()
    targets = M.split_targets(splits, SPLIT)
    runs: dict[str, dict] = {}
    for run in a.runs:
        rd = f"{M.ROOT}/runs/{run}"
        sj_path = f"{rd}/select.json"
        sj = json.load(open(sj_path)) if os.path.exists(sj_path) else None
        if run in overrides:
            epoch, source = overrides[run], "override"
            print(f"NOTE: {run}: epoch {epoch} by --epoch override"
                  + (f" (select.json chose {sj['chosen']})" if sj else " (no select.json)"), flush=True)
        else:
            if sj is None:
                die(f"{sj_path} missing; run select_epoch.py select --run {run} (or pass --epoch {run}=N)")
            if sj.get("chosen") is None:
                die(f"{sj_path}: no eligible epoch was chosen; pass an explicit --epoch {run}=N and justify it")
            if sj["split_hash"] != splits["log"]["split_hash"]:
                die(f"{sj_path} was made on split {sj['split_hash']}, splits.json is {splits['log']['split_hash']}")
            if sj.get("select_ref") != a.ref:
                print(f"WARNING: {run} was selected with reference {sj.get('select_ref')} but final eval uses {a.ref}", flush=True)
            epoch, source = int(sj["chosen"]), "select.json"
        adapter = f"{rd}/epoch_{epoch:02d}"
        if not os.path.exists(f"{adapter}/adapter_config.json"):
            die(f"no adapter at {adapter}")
        tag = f"{run}_best"
        runs[run] = {"epoch": epoch, "adapter": adapter, "tag": tag, "source": source,
                     "run_names": [f"FT_{tag}_{a.ref}_{SPLIT}_s{s}" for s in a.seeds],
                     "v2_names": [v2_name(tag, s) for s in a.seeds], "v2_dirs": [f"{M.OUT_V2}/{v2_name(tag, s)}" for s in a.seeds]}

    os.makedirs(FINAL, exist_ok=True)
    out = f"{FINAL}/plan_test.json"
    cmd = [sys.executable, BUILD_PLAN, "--out", out, "--split", SPLIT, "--ref", a.ref, "--ref-json", a.ref_json,
           "--seeds", *map(str, a.seeds)]
    for run, r in runs.items():
        cmd += ["--adapter", f"{r['tag']}={r['adapter']}"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    plan = json.load(open(out))
    fp = {"split": SPLIT, "ref": a.ref, "seeds": a.seeds, "plan_file": out, "split_hash": splits["log"]["split_hash"],
          "n_sentences": len(targets), "runs": runs, "baselines": BASELINES, "created": time.strftime("%F %T")}
    json.dump(fp, open(f"{FINAL}/final_plan.json", "w"), indent=1)

    units = len(plan["units"])
    n_clips = sum(len(u["prompts"]) for u in plan["units"])
    ntasks = min(6, units)
    half = max(1, ntasks // 2)
    print(f"\n{units} units, {n_clips} clips ({len(runs)} run(s) x {len(a.seeds)} seeds x {len(targets)} sentences) -> {out}")
    print(f"""
Next (orchestrator: pick accounts/partitions by live queue state, spread over the 5 accounts):
  cd {M.ROOT}
  # generation: same NTASKS in every submission, one --array range per account
  sbatch --account=<ACCT_A> -p <PART> --array=0-{half - 1} --export=ALL,PLAN={out},NTASKS={ntasks} final_gen.sbatch
  sbatch --account=<ACCT_B> -p <PART> --array={half}-{ntasks - 1} --export=ALL,PLAN={out},NTASKS={ntasks} final_gen.sbatch
  # then (after the generation jobs finish):
  python final_eval.py items --chunks 3          # PathBench items + final/submit_v2_ft.sh
  DRY=1 bash {FINAL}/submit_v2_ft.sh             # shows the v2 sbatch lines; then run it per run name with SBATCH_ACCOUNT=<acct>
  sbatch --account=<ACCT> -p <PART> --array=0-2 --export=ALL,FINAL={FINAL} final_score.sbatch
  sbatch --account=<ACCT> -p <PART> --array=0-2 --time=01:15:00 --mem=64G --export=ALL,FINAL={FINAL},DARTP=1 final_score.sbatch   # DArtP last, lowest weight
  python final_eval.py table""")


# ============================================================================= items
V2_HEADER = """# Generated by final_eval.py items. A copy of {src} whose TABLE holds only the finetuned runs, so the shared
# script is not edited. Same usage:  bash submit_v2_ft.sh [name ...]      DRY=1 bash submit_v2_ft.sh   (prints sbatch lines)
# Spread over accounts: SBATCH_ACCOUNT=<acct> SBATCH_PARTITION=<part> bash submit_v2_ft.sh <name>   (one run name per call)
"""


def write_v2_wrapper(fp: dict, path: str) -> str:
    src = open(SUBMIT_ALL).read()
    rows = [f"{n}|{rn}|1|1||{M.GEN}" for r in fp["runs"].values() for n, rn in zip(r["v2_names"], r["run_names"])]
    body = "\n".join(rows)
    new, k = re.subn(r"(TABLE=\$\(cat <<'EOF'\n).*?(\nEOF\n\))", lambda m: m.group(1) + body + m.group(2), src, flags=re.S)
    if k != 1:
        die(f"could not find the TABLE heredoc in {SUBMIT_ALL}; add these rows to a copy by hand:\n{body}")
    lines = new.split("\n")
    new = "\n".join([lines[0], *V2_HEADER.format(src=SUBMIT_ALL).rstrip("\n").split("\n"), *lines[1:]])
    open(path, "w").write(new)
    os.chmod(path, 0o755)
    return body


def cmd_items(a) -> None:
    fp = load_plan()
    splits = M.load_splits()
    targets = M.split_targets(splits, fp["split"])
    items = M.real_items(targets, splits)
    groups = {"real": len(items)}
    problems = []
    names = [(rn, rn) for r in fp["runs"].values() for rn in r["run_names"]] + [(b, b) for b in fp["baselines"]]
    for run_name, label in names:
        c = M.run_completeness(run_name, targets)
        if not c["ok"]:
            problems.append(f"{run_name}: {c['n_present']}/{c['n_targets']} sentences (missing audio {len(c['missing_audio'])})")
            continue
        rows = M.synth_items(run_name, label, targets)
        items += rows
        groups[label] = len(rows)
    if problems and not a.allow_incomplete:
        die("incomplete generation:\n  " + "\n  ".join(problems) + "\n(check the gen logs for FAILED lines)")
    # paired on identical text: every synthetic group covers exactly the target sentences
    for g, n in groups.items():
        if g != "real" and n != len(targets):
            print(f"WARNING: {g}: {n}/{len(targets)} sentences", flush=True)

    os.makedirs(FINAL, exist_ok=True)
    M.write_csv(f"{FINAL}/pb_items_all.csv", items, M.ITEM_FIELDS)
    for i in range(a.chunks):
        M.write_csv(f"{FINAL}/pb_items_{i:02d}.csv", items[i :: a.chunks], M.ITEM_FIELDS)
    rows = write_v2_wrapper(fp, f"{FINAL}/submit_v2_ft.sh")
    print(f"{len(items)} PathBench items in {a.chunks} chunk(s): {FINAL}/pb_items_NN.csv")
    print("groups:", json.dumps(groups))
    print(f"\nv2 table rows used in {FINAL}/submit_v2_ft.sh:\n{rows}")
    print(f"""
Commands (orchestrator submits; spread over accounts):
  DRY=1 bash {FINAL}/submit_v2_ft.sh                                  # check the sbatch lines first
  SBATCH_ACCOUNT=<acct_i> bash {FINAL}/submit_v2_ft.sh <v2 name>       # prosody + neural per run/seed -> {M.OUT_V2}/<name>/
  sbatch --account=<acct> -p <part> --array=0-{a.chunks - 1} --export=ALL,FINAL={FINAL} final_score.sbatch
  sbatch --account=<acct> -p <part> --array=0-{a.chunks - 1} --time=01:15:00 --mem=64G --export=ALL,FINAL={FINAL},DARTP=1 final_score.sbatch
  python final_eval.py table      # needs final/pb_out/*, and tts_eval/out_v2/ft_*/neural.shard* (the table reads embeddings, not the v2 summaries)""")


# ============================================================================= table
def evaluate_rule(recs: list[dict], real: dict, base_rows: dict[str, dict], conds: dict[str, dict]) -> dict:
    per_real, artp_real = real["per"], real["artp"]
    per = [r["per"] for r in recs]
    gaps = [abs(p - per_real) for p in per]
    mean_per = float(np.mean(per))
    cond_gap = {k: abs(c["per"] - per_real) for k, c in conds.items()}
    best = min(cond_gap, key=cond_gap.get)
    spread = max(per) - min(per)
    sd = float(np.std(per, ddof=1)) if len(per) > 1 else float("nan")
    dist_best = abs(mean_per - conds[best]["per"])
    improvement = cond_gap[best] - float(np.mean(gaps))
    tit = float(np.mean([r["titanet"] for r in recs]))
    within = float(np.mean([r["dur_within20"] for r in recs]))
    within_strict = float(np.mean([r["dur_within20_strict"] for r in recs]))
    ramble = float(np.mean([r["dur_ramble"] for r in recs]))
    min_base = min(base_rows, key=lambda k: base_rows[k]["dur_ramble"])
    min_ramble = base_rows[min_base]["dur_ramble"]
    artp = float(np.mean([r["artp"] for r in recs]))
    c = {
        "C1a": {"pass": bool(np.mean(gaps) <= T_PER_GAP), "value": float(np.mean(gaps)), "threshold": T_PER_GAP,
                "text": f"PER gap: mean over seeds of |synth - real| = {np.mean(gaps):.3f} <= {T_PER_GAP:.2f}",
                "detail": f"real {per_real:.3f}; seeds {[round(p, 3) for p in per]}; |mean PER - real| = {abs(mean_per - per_real):.3f}"},
        "C1b": {"pass": bool(spread < dist_best), "value": float(spread), "threshold": float(dist_best),
                "text": f"seed spread (range) {spread:.3f} < gap to best baseline {dist_best:.3f}",
                "detail": (f"best zero-shot baseline = {best} (PER {conds[best]['per']:.3f}, gap to real {cond_gap[best]:.3f}); "
                           f"FT mean PER {mean_per:.3f}; seed sd {sd:.3f}; FT improvement over it in |gap to real| = {improvement:+.3f}; "
                           f"with sd instead of range: {'pass' if sd < dist_best else 'fail'}")},
        "C2": {"pass": bool(tit >= T_TITANET), "value": tit, "threshold": T_TITANET,
               "text": f"TitaNet {tit:.3f} >= {T_TITANET:.2f}",
               "detail": f"seeds {[round(r['titanet'], 3) for r in recs]}; real ceiling {real['titanet']:.3f}"},
        "C3a": {"pass": bool(within >= T_WITHIN), "value": within, "threshold": T_WITHIN,
                "text": f"within +-20% of real length {100 * within:.0f}% >= {100 * T_WITHIN:.0f}%",
                "detail": f"seeds {[round(100 * r['dur_within20']) for r in recs]}%; strict (ratio 0.8-1.2): {100 * within_strict:.0f}%"},
        "C3b": {"pass": bool(ramble < min_ramble), "value": ramble, "threshold": min_ramble,
                "text": f"ramble rate {100 * ramble:.1f}% < every zero-shot baseline (lowest: {min_base} {100 * min_ramble:.1f}%)",
                "detail": "seeds " + str([f"{100 * r['dur_ramble']:.1f}%" for r in recs]) + "; baselines "
                          + ", ".join(f"{k} {100 * v['dur_ramble']:.1f}%" for k, v in base_rows.items())
                          + ("; NOTE: a 0% baseline makes 'strictly below' unattainable" if min_ramble <= 0 else "")
                          + f"; non-strict (<=) reading: {'pass' if ramble <= min_ramble else 'fail'}"},
        "C4": {"pass": bool(abs(artp - artp_real) <= T_ARTP), "value": float(abs(artp - artp_real)), "threshold": T_ARTP,
               "text": f"ArtP |{artp:.3f} - {artp_real:.3f}| = {abs(artp - artp_real):.3f} <= {T_ARTP:.2f}",
               "detail": f"seeds {[round(r['artp'], 3) for r in recs]}"},
    }
    missing = [f"{k} of {r['label']}" for r in recs for k in ("per", "artp", "titanet", "dur_within20", "dur_ramble")
               if not np.isfinite(r[k])]
    missing += [f"{k} of {name}" for name, rr in (("real", real), *conds.items()) for k in ("per", "artp")
                if not np.isfinite(rr[k])]
    missing += [f"dur_ramble of {k}" for k, v in base_rows.items() if not np.isfinite(v["dur_ramble"])]
    ok = all(v["pass"] for v in c.values())
    if missing:
        overall = "INCOMPLETE (missing: " + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "") + ")"
    elif len(recs) < 3:
        overall = "INCONCLUSIVE (needs 3 seeds)"
    else:
        overall = "PASS" if ok else "FAIL"
    return {"criteria": c, "n_seeds": len(recs), "overall": overall}


def _cells(rec: dict, with_dartp: bool, sd: dict | None = None) -> list[str]:
    def pm(k, d=3):
        if sd and np.isfinite(sd[k]["std"]) and sd[k]["n"] > 1:
            return f"{rec[k]:.{d}f} ± {sd[k]['std']:.{d}f}"
        return M.fmt(rec[k], d)
    cells = [pm("per"), pm("artp"), pm("titanet"), pm("wavlm"), M.fmt(rec["dur_median"], 2),
             M.fmt(rec["dur_within20"], pct=True), f"{M.fmt(rec['dur_ramble'], pct=True)}"]
    if with_dartp:
        cells.append(pm("dartp"))
    return cells


def cmd_table(a) -> None:
    fp = load_plan()
    splits = M.load_splits()
    targets = M.split_targets(splits, fp["split"])
    pb_paths = M.pb_files(f"{FINAL}/pb_out", f"{FINAL}/pb_out_dartp")
    if not pb_paths:
        die(f"no scored PathBench csv under {FINAL}/pb_out*")
    pb = M.load_pb_scored(pb_paths)
    dirs = [d for r in fp["runs"].values() for d in r["v2_dirs"]] + [f"{M.OUT_V2}/gen_{b}" for b in fp["baselines"]]
    for d in dirs + [M.REAL_BANK_DIR]:
        if not os.path.isdir(d):
            die(f"{d} missing (v2 neural scoring not run for it?)")
    store = M.EmbStore(dirs)
    bank = M.EmbStore([M.REAL_BANK_DIR])

    first_run = next(iter(fp["runs"].values()))["run_names"][0]
    real = M.score_real(targets, pb, bank, M.references_excluded(first_run))
    base = {b: M.score_run(b, b, targets, pb, store, bank) for b in fp["baselines"]}
    conds = {}
    for k, members in CONDITIONS.items():
        recs = [base[m] for m in members if m in base]
        if recs:
            conds[k] = M.mean_records(recs) if len(recs) > 1 else recs[0]
    ft = {run: [M.score_run(rn, rn, targets, pb, store, bank) for rn in r["run_names"]] for run, r in fp["runs"].items()}

    with_dartp = any(np.isfinite(r.get("dartp", float("nan"))) for rs in ft.values() for r in rs) or \
        any(np.isfinite(b["dartp"]) for b in base.values())
    head = ["Row", "PER", "ArtP", "TitaNet", "WavLM", "dur ratio (median)", "within ±20%", "ramble (>2x)"] + (["DArtP"] if with_dartp else [])
    n_s, n_c = len(targets), real["n_clips"]

    verdicts = {run: evaluate_rule(recs, real, base, conds) for run, recs in ft.items()}

    md = [f"# Final evaluation: tts_test ({n_s} sentences, {n_c} real clips)", "",
          f"Generated by `final_eval.py table` from the scored csvs; nothing typed by hand. Every row is restricted to the same "
          f"{n_s} held-out sentences and paired on identical text. Seeds {fp['seeds']}; reference `{fp['ref']}` "
          "(built only from tts_train clips). FT cells are the mean ± sd over the seeds (n=3, so sd is itself rough). "
          f"Voice match = cosine to the real-clip centroid with each run's reference sources left out of the pool; the real "
          f"row shows the real-vs-real leave-one-out ceiling. Within ±20% = synthetic/real length in [{M.WITHIN_LO}, {M.WITHIN_HI}] "
          f"(compare_variants.py definition); ramble = share of sentences longer than {M.RAMBLE_X:g}x the real clip. "
          "Real PER/ArtP are sentence-weighted means over the real repeats. DArtP is reported but weighted lowest.", "",
          "## Summary", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md.append("| **real** | " + " | ".join([M.fmt(real["per"]), M.fmt(real["artp"]), M.fmt(real["titanet"]) + " (ceiling)",
                                          M.fmt(real["wavlm"]) + " (ceiling)", "1.00", "100%", "0%"]
                                         + ([M.fmt(real["dartp"])] if with_dartp else [])) + " |")
    csv_rows = []

    def csv_row(kind, name, rec, seeds=None):
        r = {"kind": kind, "name": name, "n_sentences": n_s}
        for k in ("per", "artp", "dartp", "titanet", "wavlm", "dur_median", "dur_geo", "dur_within20", "dur_within20_strict",
                  "dur_ramble", "dur_n_ramble", "dur_max"):
            r[k] = rec.get(k, float("nan"))
        r.update({k: "" for k, v in r.items() if isinstance(v, float) and not np.isfinite(v)})
        if seeds:
            for k in ("per", "artp", "titanet", "wavlm", "dur_ramble", "dur_within20"):
                s = M.seed_summary([x[k] for x in seeds])
                r.update({f"{k}_sd": s["std"], f"{k}_min": s["min"], f"{k}_max": s["max"]})
                r.update({f"{k}_{t}": "" for t in ("sd", "min", "max") if isinstance(r[f"{k}_{t}"], float) and not np.isfinite(r[f"{k}_{t}"])})
            r["n_seeds"] = len(seeds)
        csv_rows.append(r)

    csv_row("real", "real", real)
    for run, recs in ft.items():
        mean = M.mean_records(recs)
        sd = {k: M.seed_summary([r[k] for r in recs]) for k in ("per", "artp", "titanet", "wavlm", "dartp")}
        md.append(f"| **FT {run}_best** (epoch {fp['runs'][run]['epoch']}, {len(recs)}-seed mean) | " + " | ".join(_cells(mean, with_dartp, sd)) + " |")
        csv_row("ft_mean", f"{run}_best", mean, recs)
    for k, c in conds.items():
        label = f"zero-shot {k}" + (" (3-seed mean)" if len(CONDITIONS[k]) > 1 else f" ({CONDITIONS[k][0]})")
        sd = {m: M.seed_summary([base[x][m] for x in CONDITIONS[k]]) for m in ("per", "artp", "titanet", "wavlm", "dartp")} \
            if len(CONDITIONS[k]) > 1 else None
        md.append(f"| {label} | " + " | ".join(_cells(c, with_dartp, sd)) + " |")
        csv_row("baseline_cond", k, c, [base[x] for x in CONDITIONS[k]] if len(CONDITIONS[k]) > 1 else None)

    md += ["", "## Per seed / per run", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run, recs in ft.items():
        for s, r in zip(fp["seeds"], recs):
            md.append(f"| FT {run}_best seed {s} | " + " | ".join(_cells(r, with_dartp)) + " |")
            csv_row("ft_seed", f"{run}_best_s{s}", r)
    for b in fp["baselines"]:
        md.append(f"| {b} | " + " | ".join(_cells(base[b], with_dartp)) + " |")
        csv_row("baseline", b, base[b])

    md += ["", "## Pre-registered success rule", ""]
    for run, v in verdicts.items():
        md += [f"### {run}_best (epoch {fp['runs'][run]['epoch']}): **{v['overall']}**", ""]
        for key, cr in v["criteria"].items():
            md.append(f"- **{'PASS' if cr['pass'] else 'FAIL'}** {key}: {cr['text']}  \n  _{cr['detail']}_")
        md.append("")
    md += ["Reading of the wording (all three are choices; the alternative is printed in each detail line): the PER gap is the "
           "mean of per-seed absolute gaps; seed spread is max-min over seeds; 'gap to the best baseline' is the distance between "
           "the FT mean PER and the PER of the baseline closest to real; 'below every baseline' is strict, against each of the six "
           "listed zero-shot runs.", ""]
    for run, recs in ft.items():
        miss = [r["label"] for r in recs if r["n_synth"] < r["n_targets"]]
        if miss:
            md.append(f"WARNING: incomplete generation in {miss}")
    open(f"{FINAL}/final_table.md", "w").write("\n".join(md) + "\n")
    fields = ["kind", "name", "n_sentences", "n_seeds", "per", "per_sd", "per_min", "per_max", "artp", "artp_sd", "artp_min",
              "artp_max", "dartp", "titanet", "titanet_sd", "titanet_min", "titanet_max", "wavlm", "wavlm_sd", "dur_median",
              "dur_geo", "dur_within20", "dur_within20_strict", "dur_ramble", "dur_n_ramble", "dur_max", "dur_ramble_sd",
              "dur_ramble_min", "dur_ramble_max", "dur_within20_sd", "dur_within20_min", "dur_within20_max", "wavlm_min", "wavlm_max"]
    M.write_csv(f"{FINAL}/final_table.csv", csv_rows, fields)
    json.dump(M.strip_private({"split": fp["split"], "n_sentences": n_s, "seeds": fp["seeds"], "thresholds": {
        "per_gap": T_PER_GAP, "titanet": T_TITANET, "within20": T_WITHIN, "artp_gap": T_ARTP}, "real": real,
        "verdicts": verdicts, "ft": {run: recs for run, recs in ft.items()}, "baselines": base, "conditions": conds}),
        open(f"{FINAL}/success_rule.json", "w"), indent=1)

    print("\n".join(md))
    print(f"\nwrote {FINAL}/final_table.md, final_table.csv, success_rule.json")


# ============================================================================= main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--epoch", action="append", help="RUN=N explicit epoch (needed only when select.json has none)")
    p.add_argument("--ref", default=DEFAULT_REF)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--ref-json", default=f"{M.ROOT}/refs/refs_ft.json")
    p.set_defaults(fn=cmd_plan)
    i = sub.add_parser("items")
    i.add_argument("--chunks", type=int, default=1, help="split the PathBench items into N csvs (one final_score array task each)")
    i.add_argument("--allow-incomplete", action="store_true")
    i.set_defaults(fn=cmd_items)
    t = sub.add_parser("table")
    t.set_defaults(fn=cmd_table)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
