"""Phase 3: choose the LoRA checkpoint (epoch) on tts_val by generation quality, never by loss.

    python select_epoch.py plan   --run A_r16                 # generation plan for epochs {1,2,3,5,8,12,15} (+ pretrained)
    python select_epoch.py score  --run A_r16                 # prep -> voice (GPU) -> PathBench -> aggregate
    python select_epoch.py select --run A_r16                 # apply the pre-registered rule, write select.md / select.json

Flow (see select_gen.sbatch / select_score.sbatch; accounts and partitions go on the sbatch command line):

  1. plan    run on the login node (light: only writes manifests via build_plan_ft.py). Writes
             runs/<run>/select/{plan_val.json,select_plan.json}; run folders land in gen_runs/
             FT_<run>_eNN_<ref>_tts_val_s<seed>/, paired by prompt id with the zero-shot baselines.
  2. gen     gen_ft.py --plan runs/<run>/select/plan_val.json (select_gen.sbatch, GPU array).
  3. score   prep   check every run is complete, write the PathBench items csv (real val clips + every epoch +
                    the zero-shot stitch10 baseline R1b_stitch10_s1 on the SAME sentences);
             voice  TitaNet + WavLM embeddings of the generated clips via tts_eval/neural_metrics_v2.Scorer (GPU);
             pb     pb_eval/pb_score_items.py (no --dartp) in the PathBench env;
             aggregate  ft_metrics: duration ratio / within +-20% / ramble rate, PER, ArtP, TitaNet, WavLM.
  4. select  rule fixed BEFORE looking at results (plan section "Phase 3"):

       lowest |PER_synth - PER_real|  subject to
         ramble rate  <= the zero-shot stitch10 ramble rate on the same sentences, and
         TitaNet voice match >= 0.60.

     Ties go to the earlier epoch. The pretrained model with the same reference (epoch 0) and any extra references
     are reported in the table but are never selected.

tts_val has only 20 sentences: a PER difference below ~0.05 is noise (see the paired SE printed next to every gap).
The rule is a coarse screen against collapse / rambling, not a fine ranking; select.md lists which eligible epochs
are within the noise of the chosen one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_metrics as M  # noqa: E402

EPOCHS = [1, 2, 3, 5, 8, 12, 15]
SPLIT = "tts_val"
BASELINE = "R1b_stitch10_s1"        # zero-shot stitch10, seed 1: ramble-rate yardstick and table comparator
DEFAULT_REF = "ft_stitch5"
MIN_TITANET = 0.60
NOISE_PER = 0.05
BUILD_PLAN = os.environ.get("BUILD_PLAN_FT", os.path.join(HERE, "build_plan_ft.py"))
PB_ENV = f"{M.PB_EVAL}/pb_env.sh"
RULE_TEXT = ("lowest |PER_synth - PER_real| subject to ramble rate <= zero-shot stitch10 ramble rate (same sentences) "
             f"and TitaNet voice match >= {MIN_TITANET:.2f}; ties -> earlier epoch")


def rdir(run: str) -> str:
    return f"{M.ROOT}/runs/{run}"


def sdir(run: str) -> str:
    return f"{rdir(run)}/select"


def banner(n_sent: int, n_clips: int | None = None) -> str:
    clips = f" ({n_clips} real clips)" if n_clips else ""
    line = "!" * 100
    return (f"\n{line}\n!!  tts_val has only {n_sent} sentences{clips}: n={n_sent}. PER differences below ~{NOISE_PER:.2f} "
            f"are NOISE.\n!!  Treat this as a screen for collapse/rambling/voice loss, not a fine ranking of epochs.\n{line}\n")


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def load_meta(run: str) -> dict:
    p = f"{sdir(run)}/select_plan.json"
    if not os.path.exists(p):
        die(f"{p} missing; run `select_epoch.py plan --run {run}` first")
    return json.load(open(p))


# ============================================================================= plan
def cmd_plan(a) -> None:
    refs_all = json.load(open(a.ref_json))
    refs_all = refs_all.get("refs") or refs_all
    for r in a.refs:
        if r not in refs_all:
            die(f"reference '{r}' not in {a.ref_json} (has: {sorted(refs_all)})")

    adapters, missing = [], []
    for e in a.epochs:
        path = f"{rdir(a.run)}/epoch_{e:02d}"
        if os.path.exists(f"{path}/adapter_config.json"):
            adapters.append((f"{a.run}_e{e:02d}", path, e))
        else:
            missing.append(e)
    if missing:
        msg = f"no adapter at {rdir(a.run)}/epoch_NN for epochs {missing}"
        if not a.skip_missing:
            die(msg + " (training not finished? use --skip-missing to plan the rest)")
        print(f"WARNING: {msg}; skipped", flush=True)
    if not adapters:
        die("no epoch adapters found")
    if not a.no_pretrained:
        adapters.append(("pretrained", "none", 0))   # same reference, no adapter: the like-for-like zero-shot control

    sd = sdir(a.run)
    os.makedirs(sd, exist_ok=True)
    units, entries, model_path = [], [], None
    for ref in a.refs:
        out = f"{sd}/plan_val_{ref}.json"
        cmd = [sys.executable, BUILD_PLAN, "--out", out, "--split", SPLIT, "--ref", ref, "--ref-json", a.ref_json,
               "--seeds", *map(str, a.seeds)]
        for tag, path, _ in adapters:
            cmd += ["--adapter", f"{tag}={path}"]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        plan = json.load(open(out))
        model_path = plan["model_path"]
        units += plan["units"]
        for tag, path, e in adapters:
            for s in a.seeds:
                entries.append({"epoch": e, "tag": tag, "ref": ref, "seed": s, "adapter": path,
                                "run_name": f"FT_{tag}_{ref}_{SPLIT}_s{s}"})
    plan_path = f"{sd}/plan_val.json"
    json.dump({"model_path": model_path, "units": units}, open(plan_path, "w"), indent=1)
    splits = M.load_splits()
    targets = M.split_targets(splits, SPLIT)
    meta = {"run": a.run, "split": SPLIT, "refs": a.refs, "seeds": a.seeds, "epochs": [e for _, _, e in adapters],
            "plan_file": plan_path, "entries": entries, "n_sentences": len(targets),
            "split_hash": splits["log"]["split_hash"], "baseline": BASELINE, "created": time.strftime("%F %T")}
    json.dump(meta, open(f"{sd}/select_plan.json", "w"), indent=1)

    n_clips = sum(len(u["prompts"]) for u in units)
    ntasks = min(4, len(units))
    half = max(1, ntasks // 2)
    first = f"--array=0-{half - 1}" if ntasks > 1 else "--array=0-0"
    second = (f"  sbatch --account=<ACCT_B> -p <PART> --array={half}-{ntasks - 1} "
              f"--export=ALL,PLAN={plan_path},NTASKS={ntasks} select_gen.sbatch\n") if ntasks > 1 else ""
    print(banner(len(targets), sum(t["n_real"] for t in targets.values())))
    print(f"{len(units)} units, {n_clips} clips, {len(entries)} run folders -> {plan_path}")
    print(f"""
Next (orchestrator: pick accounts/partitions by live queue state, spread over the 5 accounts):
  cd {M.ROOT}
  # generation, ~{n_clips} clips at roughly 3-6 s each. The array index is the stripe of units and NTASKS must be the
  # same in every submission, so one array can be split across accounts (one --array range per account):
  sbatch --account=<ACCT_A> -p <PART> {first} --export=ALL,PLAN={plan_path},NTASKS={ntasks} select_gen.sbatch
{second}  # scoring + selection once every generation task has finished (--dependency=afterok:<jobids>):
  sbatch --account=<ACCT> -p <PART> --dependency=afterok:<gen jobids> --export=ALL,RUN={a.run} select_score.sbatch
  # -> {rdir(a.run)}/select.md and select.json""")


# ============================================================================= score
def stage_prep(a, meta: dict) -> None:
    splits = M.load_splits()
    targets = M.split_targets(splits, SPLIT)
    if splits["log"]["split_hash"] != meta["split_hash"]:
        die(f"splits.json changed since plan ({splits['log']['split_hash']} vs {meta['split_hash']}); re-plan")
    bad = []
    for name in [e["run_name"] for e in meta["entries"]] + [BASELINE]:
        c = M.run_completeness(name, targets)
        if not c["ok"]:
            bad.append(f"{name}: {c['n_present']}/{c['n_targets']} sentences (missing {len(c['missing'])}, "
                       f"missing audio {len(c['missing_audio'])})")
        elif name != BASELINE and c["n_rows"] != len(targets):   # the baseline run holds all 308 sentences
            print(f"  note: {name} has {c['n_rows']} metadata rows for {len(targets)} sentences", flush=True)
    if bad:
        msg = "incomplete generation:\n  " + "\n  ".join(bad)
        if not a.allow_incomplete:
            die(msg + "\n(check the gen logs for FAILED lines; --allow-incomplete scores what exists)")
        print("WARNING: " + msg, flush=True)

    items = M.real_items(targets, splits)
    for e in meta["entries"]:
        items += M.synth_items(e["run_name"], e["run_name"], targets)
    items += M.synth_items(BASELINE, BASELINE, targets)
    sd = sdir(a.run)
    M.write_csv(f"{sd}/pb_items.csv", items, M.ITEM_FIELDS)
    todo = []
    for e in meta["entries"]:
        by, _ = M.load_synth(e["run_name"])
        todo += [by[n]["audio_path"] for n in targets if n in by]
    json.dump(sorted(set(todo)), open(f"{sd}/voice_todo.json", "w"))
    print(f"prep: {len(items)} PathBench items ({sum(1 for i in items if i['source'] == 'real')} real) -> {sd}/pb_items.csv; "
          f"{len(set(todo))} clips to embed", flush=True)


def make_scorer():
    """neural_metrics_v2.Scorer (HuBERT + WavLM-SV + TitaNet) and its 16 kHz loader; GPU if there is one."""
    sys.path.insert(0, M.TTS_EVAL)
    import neural_metrics_v2 as nm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU visible; embedding on CPU (slow)", flush=True)
    return nm.Scorer(device), nm._load


def stage_voice(a, meta: dict) -> None:
    sd = sdir(a.run)
    out = f"{sd}/voice"
    todo = json.load(open(f"{sd}/voice_todo.json"))
    rows, emb = [], {m: [] for m in M.SV_MODELS}
    have = set()
    if os.path.exists(f"{out}/neural.shard000of001.csv") and not a.force:
        old_rows, old = M.load_shards(out)
        rows = old_rows
        emb = {m: [old[m][i] for i in range(len(rows))] for m in M.SV_MODELS}
        have = {r["audio_path"] for r in rows if not r["error"]}
    new = [p for p in todo if p not in have]
    print(f"voice: {len(have)} clips already embedded, {len(new)} to do", flush=True)
    if new:
        scorer, load = make_scorer()
        for i, p in enumerate(new, 1):
            row = {k: "" for k in M.SHARD_FIELDS}
            row.update(group="synthetic", utt_id=f"v{len(rows)}", audio_path=p, speaker="bingge")
            try:
                e = scorer.embed(load(p))
                for m in M.SV_MODELS:
                    emb[m].append(np.asarray(e[m], dtype=np.float32))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                for m, d in (("wavlm", 512), ("titanet", 192)):
                    emb[m].append(np.full(d, np.nan, dtype=np.float32))
            rows.append(row)
            if i % 25 == 0:
                print(f"  embedded {i}/{len(new)}", flush=True)
        os.makedirs(out, exist_ok=True)
        M.write_csv(f"{out}/neural.shard000of001.csv", rows, M.SHARD_FIELDS)
        np.savez(f"{out}/neural.shard000of001.npz", **{m: np.stack(v) for m, v in emb.items()})
    print(f"voice: wrote {out} ({sum(1 for r in rows if r['error'])} errors)", flush=True)


def stage_pb(a, meta: dict) -> None:
    sd = sdir(a.run)
    items, outd = f"{sd}/pb_items.csv", f"{sd}/pb_out"
    scored = f"{outd}/items_scored.csv"
    n_items = len(M.read_csv(items))
    if os.path.exists(scored) and not a.force and len(M.read_csv(scored)) >= n_items:
        print(f"pb: {scored} already covers {n_items} items; skipping (--force to redo)", flush=True)
        return
    # PathBench needs its own site dir (pb_env.sh); no --dartp for selection (cheap evaluators only)
    cmd = (f'source "{PB_ENV}" && cd "{M.PB_EVAL}" && export HF_HUB_OFFLINE=1 && '
           f'"$PB_PY" pb_score_items.py --items "{items}" --out-dir "{outd}"')
    print("+ bash -c '" + cmd + "'", flush=True)
    if a.dry_run:
        return
    subprocess.run(["bash", "-c", cmd], check=True)


def aggregate(run: str) -> dict:
    meta = load_meta(run)
    sd = sdir(run)
    splits = M.load_splits()
    targets = M.split_targets(splits, SPLIT)
    pb_paths = M.pb_files(f"{sd}/pb_out")
    if not pb_paths:
        die(f"no PathBench scores under {sd}/pb_out (run `score --stage pb`)")
    pb = M.load_pb_scored(pb_paths)
    for d in (f"{sd}/voice", f"{M.OUT_V2}/gen_{BASELINE}", M.REAL_BANK_DIR):
        if not os.path.isdir(d):
            die(f"{d} missing (voice stage not run? baseline v2 output moved?)")
    store = M.EmbStore([f"{sd}/voice", f"{M.OUT_V2}/gen_{BASELINE}"])
    bank = M.EmbStore([M.REAL_BANK_DIR])

    real = M.score_real(targets, pb, bank, M.references_excluded(meta["entries"][0]["run_name"]))
    base = M.score_run(BASELINE, BASELINE, targets, pb, store, bank)

    groups: dict[tuple[int, str], list[dict]] = {}
    for e in meta["entries"]:
        rec = M.score_run(f"{e['tag']}@{e['ref']}", e["run_name"], targets, pb, store, bank)
        rec.update(epoch=e["epoch"], ref=e["ref"], tag=e["tag"], seed=e["seed"], adapter=e["adapter"])
        groups.setdefault((e["epoch"], e["ref"]), []).append(rec)

    epochs = []
    for (ep, ref), recs in sorted(groups.items()):
        r = M.mean_records(recs)
        sent = {}
        for k in recs[0]["_sent"]:
            names = set().union(*[set(x["_sent"][k]) for x in recs])
            sent[k] = {n: M.mean_or_nan([x["_sent"][k].get(n) for x in recs]) for n in names}
        r["_sent"] = sent
        diff = M.paired_se(sent["per"], real["_sent"]["per"])
        r["per_diff"] = diff["mean"]                       # signed: synth - real
        r["per_gap"] = abs(diff["mean"]) if diff["n"] else float("nan")
        r["per_gap_se"] = diff["se"]
        r["artp_gap"] = abs(r["artp"] - real["artp"])
        r["epoch"], r["ref"], r["tag"] = ep, ref, recs[0]["tag"]
        r["seeds"] = [x["seed"] for x in recs]
        epochs.append(r)

    out = {"run": run, "split": SPLIT, "n_sentences": len(targets), "n_real_clips": real["n_clips"],
           "split_hash": meta["split_hash"], "refs": meta["refs"], "seeds": meta["seeds"], "real": real,
           "baseline": base, "epochs": epochs, "noise_per": NOISE_PER}
    M.write_csv(f"{sd}/metrics.csv", [{k: v for k, v in e.items() if not k.startswith("_") and not isinstance(v, (list, dict))}
                                      for e in [{**real, "epoch": "real", "ref": "", "tag": "real"},
                                                {**base, "epoch": "zero-shot", "ref": "stitch10", "tag": BASELINE}] + epochs],
                ["epoch", "ref", "tag", "n_targets", "n_synth", "per", "per_gap", "per_gap_se", "artp", "artp_gap", "dartp",
                 "titanet", "wavlm", "dur_median", "dur_geo", "dur_within20", "dur_within20_strict", "dur_ramble",
                 "dur_n_ramble", "dur_max", "per_n", "artp_n", "titanet_n"])
    json.dump(M.strip_private(out), open(f"{sd}/metrics.json", "w"), indent=1)
    return out


def cmd_score(a) -> None:
    meta = load_meta(a.run)
    stages = ["prep", "voice", "pb", "aggregate"] if a.stage == "all" else [a.stage]
    if "prep" in stages:
        stage_prep(a, meta)
    if "voice" in stages:
        stage_voice(a, meta)
    if "pb" in stages:
        stage_pb(a, meta)
    if "aggregate" in stages:
        m = aggregate(a.run)
        print(banner(m["n_sentences"], m["n_real_clips"]))
        print(f"aggregate: wrote {sdir(a.run)}/metrics.json / metrics.csv")


# ============================================================================= select
def _row(e: dict, extra: str = "") -> list[str]:
    return [str(e.get("epoch", "")), e.get("ref", ""), M.fmt(e["per"]), M.fmt(e.get("per_gap")),
            M.fmt(e.get("per_gap_se")), M.fmt(e["artp"]), M.fmt(e["titanet"]), M.fmt(e["wavlm"]),
            M.fmt(e["dur_median"], 2), M.fmt(e["dur_within20"], pct=True),
            f"{M.fmt(e['dur_ramble'], pct=True)} ({e['dur_n_ramble']:g}/{e['dur_n']:g})", extra]


def apply_rule(metrics: dict, select_ref: str) -> dict:
    base, real = metrics["baseline"], metrics["real"]
    max_ramble = base["dur_ramble"]
    cands = [e for e in metrics["epochs"] if e["epoch"] > 0 and e["ref"] == select_ref]
    if not cands:
        die(f"no epoch entries for reference '{select_ref}' (have {sorted({e['ref'] for e in metrics['epochs']})})")
    for e in cands:
        why = []
        if e["n_synth"] < e["n_targets"]:
            why.append({"kind": "missing", "text": f"only {e['n_synth']}/{e['n_targets']} sentences generated"})
        if not np.isfinite(e["per"]):
            why.append({"kind": "missing", "text": "no PER"})
        if not np.isfinite(e["dur_ramble"]):
            why.append({"kind": "missing", "text": "no duration"})
        elif e["dur_ramble"] > max_ramble + 1e-9:
            why.append({"kind": "ramble", "text": f"ramble {e['dur_ramble']:.3f} > stitch10 {max_ramble:.3f}"})
        if not np.isfinite(e["titanet"]):
            why.append({"kind": "missing", "text": "no TitaNet"})
        elif e["titanet"] < MIN_TITANET:
            why.append({"kind": "voice", "text": f"TitaNet {e['titanet']:.3f} < {MIN_TITANET:.2f}"})
        e["eliminated_by"] = why
        e["eligible"] = not why
    elig = sorted([e for e in cands if e["eligible"]], key=lambda e: (e["per_gap"], e["epoch"]))
    chosen = elig[0] if elig else None
    return {"select_ref": select_ref, "max_ramble_rate": max_ramble, "min_titanet": MIN_TITANET, "cands": cands,
            "eligible": elig, "chosen": chosen,
            "within_noise": [e["epoch"] for e in elig[1:] if chosen and e["per_gap"] - chosen["per_gap"] <= NOISE_PER]}


def cmd_select(a) -> None:
    metrics = aggregate(a.run)
    sel_ref = a.select_ref or metrics["refs"][0]
    res = apply_rule(metrics, sel_ref)
    real, base, chosen = metrics["real"], metrics["baseline"], res["chosen"]
    n = metrics["n_sentences"]

    head = ["epoch", "ref", "PER", "|PER-real|", "gap SE", "ArtP", "TitaNet", "WavLM", "dur ratio med", "within +-20%",
            "ramble (>2x)", "verdict"]
    rows = [["real", "", M.fmt(real["per"]), "0", "", M.fmt(real["artp"]), M.fmt(real["titanet"]) + " (ceiling)",
             M.fmt(real["wavlm"]) + " (ceiling)", "1.00", "100%", "0%", f"{real['n_targets']} sent / {real['n_clips']} clips"],
            ["zero-shot stitch10", BASELINE, M.fmt(base["per"]), M.fmt(abs(base["per"] - real["per"])), "", M.fmt(base["artp"]),
             M.fmt(base["titanet"]), M.fmt(base["wavlm"]), M.fmt(base["dur_median"], 2), M.fmt(base["dur_within20"], pct=True),
             f"{M.fmt(base['dur_ramble'], pct=True)} ({base['dur_n_ramble']}/{base['dur_n']})", "ramble yardstick"]]
    for e in sorted(metrics["epochs"], key=lambda x: (x["ref"] != sel_ref, x["ref"], x["epoch"])):
        if e["epoch"] == 0:
            verdict = "pretrained, same ref (control)"
        elif e["ref"] != sel_ref:
            verdict = "other ref (info only)"
        elif chosen and e["epoch"] == chosen["epoch"] and e["ref"] == sel_ref:
            verdict = "**CHOSEN**"
        elif e.get("eligible"):
            verdict = "eligible" + (" (within noise of chosen)" if e["epoch"] in res["within_noise"] else "")
        else:
            verdict = "eliminated: " + "; ".join(w["text"] for w in e["eliminated_by"])
        rows.append(_row(e, verdict))
    md = [f"# Checkpoint selection: {a.run}", "",
          f"> **tts_val has only {n} sentences ({real['n_clips']} real clips). n={n}. PER differences below ~{NOISE_PER:.2f} are "
          f"noise;** gap SE is the standard error of the per-sentence (synth - real) PER difference. This is a screen for "
          "collapse / rambling / voice loss, not a fine ranking.", "",
          f"Rule (pre-registered): {RULE_TEXT}.",
          f"Selection reference: `{sel_ref}` (built only from tts_train clips). Seeds: {metrics['seeds']}. "
          f"Split hash {metrics['split_hash']}.",
          f"Ramble rate = share of sentences whose synthetic clip is > {M.RAMBLE_X:g}x the real length; within +-20% = ratio in "
          f"[{M.WITHIN_LO}, {M.WITHIN_HI}]. Zero-shot stitch10 ramble rate on these sentences: "
          f"{M.fmt(res['max_ramble_rate'], pct=True)} ({base['dur_n_ramble']}/{base['dur_n']}).", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md.append("")
    if chosen:
        md.append(f"**Chosen: epoch {chosen['epoch']}** (|PER-real| = {chosen['per_gap']:.3f}, ramble "
                  f"{M.fmt(chosen['dur_ramble'], pct=True)}, TitaNet {chosen['titanet']:.3f}) -> `{chosen['adapter']}`")
        if res["within_noise"]:
            md.append(f"Eligible epochs within {NOISE_PER:.2f} PER of the chosen one (not distinguishable at n={n}): "
                      f"{res['within_noise']}. The rule still picks the lowest gap; look at their audio before committing.")
    else:
        md.append("**NO ELIGIBLE EPOCH**: every epoch is eliminated by the ramble or voice constraint (see verdict column). "
                  "Nothing is chosen; final_eval.py plan needs an explicit --epoch override, which should be argued in "
                  "the write-up (e.g. tighten training, run B/C), not hidden.")
    open(f"{rdir(a.run)}/select.md", "w").write("\n".join(md) + "\n")

    sj = {"run": a.run, "split": SPLIT, "n_sentences": n, "n_real_clips": real["n_clips"], "split_hash": metrics["split_hash"],
          "select_ref": sel_ref, "seeds": metrics["seeds"], "rule": RULE_TEXT,
          "constraints": {"max_ramble_rate": res["max_ramble_rate"], "min_titanet": MIN_TITANET,
                          "ramble_yardstick": BASELINE},
          "noise_note": f"tts_val has {n} sentences; PER differences below ~{NOISE_PER:.2f} are noise",
          "real": {k: real[k] for k in ("per", "artp", "titanet", "wavlm", "n_targets", "n_clips")},
          "zero_shot_stitch10": {k: base[k] for k in ("per", "artp", "titanet", "wavlm", "dur_ramble", "dur_n_ramble",
                                                      "dur_within20", "dur_median", "dur_n")},
          "chosen": chosen["epoch"] if chosen else None,
          "chosen_adapter": chosen["adapter"] if chosen else None,
          "chosen_metrics": {k: chosen[k] for k in ("per", "per_gap", "artp", "titanet", "wavlm", "dur_ramble",
                                                    "dur_within20", "dur_median")} if chosen else None,
          "eligible_epochs": [e["epoch"] for e in res["eligible"]],
          "within_noise_of_chosen": res["within_noise"],
          "eliminated": {str(e["epoch"]): {"by": sorted({w["kind"] for w in e["eliminated_by"]}),
                                           "reasons": [w["text"] for w in e["eliminated_by"]]}
                         for e in res["cands"] if not e["eligible"]},
          "epochs": {str(e["epoch"]): {k: e[k] for k in ("per", "per_gap", "per_gap_se", "artp", "titanet", "wavlm",
                                                         "dur_median", "dur_within20", "dur_ramble", "n_synth")}
                     for e in res["cands"]},
          "created": time.strftime("%F %T")}
    json.dump(M.strip_private(sj), open(f"{rdir(a.run)}/select.json", "w"), indent=1)

    print(banner(n, real["n_clips"]))
    print("\n".join(md))
    print(f"\nwrote {rdir(a.run)}/select.md and select.json")
    if not chosen:
        sys.exit(3)


# ============================================================================= main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    p.add_argument("--refs", nargs="+", default=[DEFAULT_REF],
                   help="keys of refs_ft.json; the first is the selection reference, the others are reported only")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--ref-json", default=f"{M.ROOT}/refs/refs_ft.json")
    p.add_argument("--no-pretrained", action="store_true", help="skip the epoch-0 control (pretrained + same reference)")
    p.add_argument("--skip-missing", action="store_true")
    p.set_defaults(fn=cmd_plan)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--stage", choices=["all", "prep", "voice", "pb", "aggregate"], default="all")
    s.add_argument("--force", action="store_true")
    s.add_argument("--allow-incomplete", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="print the PathBench command instead of running it")
    s.set_defaults(fn=cmd_score)
    t = sub.add_parser("select")
    t.add_argument("--run", required=True)
    t.add_argument("--select-ref", default=None)
    t.set_defaults(fn=cmd_select)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
