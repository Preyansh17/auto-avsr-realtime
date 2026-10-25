"""Shared metric helpers for the Higgs LoRA finetune's Phase 3 (select_epoch.py) and Phase 4 (final_eval.py).

Everything here is numpy + csv only (no torch, no GPU) and works on the outputs the existing evaluation already
writes, so the same numbers are produced for a finetuned run and for a zero-shot baseline run:

  duration   per-sentence ratio synth_dur / real_dur (real_dur = mean over the real repeats of that sentence),
             median, share within +-20% (ratio in [0.8, 1.25], the definition higgs_gen/compare_variants.py
             `subset_metrics` used for the "37-46% of clips" baseline numbers), strict variant (0.8-1.2), and the
             ramble rate = share of sentences whose synthetic clip is longer than 2x the real one.
  voice      cosine of a synthetic clip's speaker embedding to the centroid of the patient's real clips, with the
             clips the reference was stitched from left out of the pool: the same arithmetic as
             tts_eval/neural_metrics_v2.py `speaker_block` / higgs_gen/compare_variants.py `load` ("sim"). The
             embeddings themselves come from neural_metrics_v2.Scorer (TitaNet + WavLM); this module only reads
             the neural.shard*.csv/.npz files that Scorer's callers write, so nothing is re-implemented.
             The real pool is a bank of ALL real clips' embeddings (default: the R1_stitch5_s1 v2 neural run, which
             holds the 394 real clips), NOT the run folder's own real manifest, because a finetuned run's manifest
             only lists its split's real clips and would give a different, noisier centroid than the baselines'.
  PathBench  per-clip scores from pb_eval/pb_score_items.py (items_scored.csv); `severity` is the group label.
             Real and synthetic means are taken per sentence (mean over the real repeats first), so each held-out
             sentence counts once on both sides.

Nothing here decides anything; the pre-registered rules live in select_epoch.py and final_eval.py.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.environ.get("HIGGS_FT_ROOT", "/scratch/pa2753/higgs_ft")
GEN = os.environ.get("HIGGS_FT_GEN", "/scratch/pa2753/gen_runs")
TTS_EVAL = os.environ.get("TTS_EVAL_DIR", "/scratch/pa2753/tts_eval")
OUT_V2 = os.environ.get("TTS_EVAL_OUT_V2", f"{TTS_EVAL}/out_v2")
PB_EVAL = os.environ.get("PB_EVAL_DIR", "/scratch/pa2753/pb_eval")
HIGGS_GEN = os.environ.get("HIGGS_GEN", "/scratch/pa2753/higgs_gen")
REAL_BANK_DIR = os.environ.get("HIGGS_FT_REAL_BANK", f"{OUT_V2}/gen_R1_stitch5_s1")

SV_MODELS = ("titanet", "wavlm")
WITHIN_LO, WITHIN_HI, WITHIN_STRICT_HI = 0.8, 1.25, 1.2   # compare_variants.subset_metrics: ratio in [0.8, 1.25]
RAMBLE_X = 2.0
PB_METRICS = ("per", "artp", "dartp", "dper", "artp_old", "cpp", "praat_rate", "std_pitch", "wada_snr")
ITEM_FIELDS = ["source", "speaker", "severity", "audio_path", "transcript"]
SHARD_FIELDS = ["group", "utt_id", "audio_path", "transcript", "speaker", "severity", "session",
                "hypothesis", "wer", "cer", "n_ref", "n_sub", "n_del", "n_ins", "error"]  # neural_metrics_v2.CSV_FIELDS


def _normalize_fallback(text: str) -> str:  # identical to higgs_gen/gen_higgs.py normalize_text
    text = text.replace("﻿", " ")
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"[^A-Za-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


try:  # prefer the original so the two cannot drift; gen_higgs imports no torch at module level
    if os.path.isdir(HIGGS_GEN) and HIGGS_GEN not in sys.path:
        sys.path.insert(0, HIGGS_GEN)
    from gen_higgs import normalize_text as norm  # noqa: E402
except Exception:  # pragma: no cover - local checkout without the cluster tree
    norm = _normalize_fallback


# ----------------------------------------------------------------------------- small utils
def fl(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def finite(vals) -> np.ndarray:
    a = np.asarray([fl(v) for v in vals], dtype=float)
    return a[np.isfinite(a)]


def mean_or_nan(vals) -> float:
    a = finite(vals)
    return float(a.mean()) if a.size else float("nan")


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(x, d: int = 3, pct: bool = False) -> str:
    x = fl(x)
    if not np.isfinite(x):
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.{d}f}"


def seed_summary(values) -> dict:
    a = finite(values)
    if not a.size:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "range": float("nan")}
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "range": float(a.max() - a.min())}


def strip_private(obj):
    """Drop keys starting with '_' (per-sentence payloads) and turn NaN into null for json."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [strip_private(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ----------------------------------------------------------------------------- splits / sentences
def load_splits(root: str | None = None) -> dict:
    return json.load(open(f"{root or ROOT}/data/splits.json"))


def split_targets(splits: dict, split: str) -> dict[str, dict]:
    """norm -> {norm, text, utt_ids, audio_paths, durs, real_dur (mean over repeats), n_real}."""
    out: dict[str, dict] = {}
    for u in splits[split]:
        c = splits["clips"][u]
        t = out.setdefault(c["norm"], {"norm": c["norm"], "text": c["transcript"], "utt_ids": [], "audio_paths": [],
                                       "durs": []})
        t["utt_ids"].append(u)
        t["audio_paths"].append(c["audio_path"])
        t["durs"].append(float(c["duration_sec"]))
    for t in out.values():
        t["real_dur"] = float(np.mean(t["durs"]))
        t["n_real"] = len(t["durs"])
    return dict(sorted(out.items()))


def real_items(targets: dict[str, dict], splits: dict) -> list[dict]:
    rows = []
    for t in targets.values():
        for u in t["utt_ids"]:
            c = splits["clips"][u]
            rows.append({"source": "real", "speaker": "bingge", "severity": "real", "audio_path": c["audio_path"],
                         "transcript": c["transcript"]})
    return rows


# ----------------------------------------------------------------------------- generated runs
def load_synth(run_name: str, gen_root: str | None = None) -> tuple[dict[str, dict], int]:
    """norm -> metadata row of a run folder's synthetic/metadata/*.csv, and the raw row count."""
    files = sorted(glob.glob(f"{gen_root or GEN}/{run_name}/synthetic/metadata/*.csv"))
    by_norm: dict[str, dict] = {}
    n = 0
    for f in files:
        for r in read_csv(f):
            n += 1
            by_norm.setdefault(r.get("norm_transcript") or norm(r["transcript"]), r)
    return by_norm, n


def synth_items(run_name: str, label: str, targets: dict[str, dict], gen_root: str | None = None) -> list[dict]:
    """PathBench items for one run, restricted to the target sentences, with the target text as transcript
    (so every group is scored against identical text)."""
    by_norm, _ = load_synth(run_name, gen_root)
    return [{"source": "synthetic", "speaker": "bingge", "severity": label, "audio_path": by_norm[n]["audio_path"],
             "transcript": t["text"]} for n, t in targets.items() if n in by_norm]


def run_completeness(run_name: str, targets: dict[str, dict], gen_root: str | None = None) -> dict:
    by_norm, n_rows = load_synth(run_name, gen_root)
    missing = sorted(set(targets) - set(by_norm))
    absent = [n for n in targets if n in by_norm and not os.path.exists(by_norm[n]["audio_path"])]
    return {"run": run_name, "n_rows": n_rows, "n_targets": len(targets), "n_present": len(targets) - len(missing),
            "missing": missing, "missing_audio": absent, "ok": not missing and not absent}


def references_excluded(run_name: str, gen_root: str | None = None) -> set[str]:
    """Real clips the run was conditioned on (reference + stitched sources); same rule as
    neural_metrics_v2._excluded_real_paths on the run's manifests/references.csv."""
    p = f"{gen_root or GEN}/{run_name}/manifests/references.csv"
    out: set[str] = set()
    if os.path.exists(p):
        for row in read_csv(p):
            out.add(row.get("audio_path", ""))
            out |= {s for s in (row.get("stitched_source_audio_paths") or "").split("|") if s}
    return out


# ----------------------------------------------------------------------------- duration
def duration_ratios(by_norm: dict[str, dict], targets: dict[str, dict]) -> dict[str, float]:
    out = {}
    for n, t in targets.items():
        s = by_norm.get(n)
        if s is None:
            continue
        sd, rd = fl(s["duration_sec"]), t["real_dur"]
        if rd > 0 and sd > 0:
            out[n] = sd / rd
    return out


def duration_stats(ratios) -> dict:
    r = finite(ratios)
    r = r[r > 0]
    nan = float("nan")
    if not r.size:
        return {"n": 0, "median": nan, "geo": nan, "logsd": nan, "within20": nan, "within20_strict": nan,
                "ramble": nan, "n_ramble": 0, "max": nan}
    lr = np.log(r)
    return {"n": int(r.size), "median": float(np.median(r)), "geo": float(np.exp(lr.mean())),
            "logsd": float(lr.std(ddof=1)) if r.size > 1 else nan,
            "within20": float(np.mean((r >= WITHIN_LO) & (r <= WITHIN_HI))),
            "within20_strict": float(np.mean((r >= WITHIN_LO) & (r <= WITHIN_STRICT_HI))),
            "ramble": float(np.mean(r > RAMBLE_X)), "n_ramble": int((r > RAMBLE_X).sum()), "max": float(r.max())}


# ----------------------------------------------------------------------------- voice (speaker embeddings)
def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def _cos(a: np.ndarray, b: np.ndarray) -> float:  # neural_metrics_v2.cosine
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def load_shards(out_dir: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    """neural.shard*.csv + .npz of a neural_metrics_v2 run (or select_epoch's voice stage, same format)."""
    files = sorted(glob.glob(os.path.join(out_dir, "neural.shard*.csv")))
    if not files:
        raise FileNotFoundError(f"no neural.shard*.csv in {out_dir}")
    rows: list[dict] = []
    embs: dict[str, list] = {m: [] for m in SV_MODELS}
    for f in files:
        part = read_csv(f)
        z = np.load(f.replace(".csv", ".npz"))
        for m in SV_MODELS:
            if len(z[m]) != len(part):
                raise ValueError(f"{f}: {len(part)} csv rows vs {len(z[m])} {m} embeddings")
            embs[m].append(z[m])
        rows += part
    return rows, {m: np.concatenate(v) for m, v in embs.items()}


class EmbStore:
    """(group, audio_path) -> {model: unit embedding} over one or more v2-style neural shard dirs."""

    def __init__(self, dirs: list[str]):
        self.vec: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for d in dirs:
            rows, embs = load_shards(d)
            unit = {m: _unit(embs[m]) for m in SV_MODELS}
            for i, r in enumerate(rows):
                if r.get("error") and not r["error"].startswith("excluded"):
                    continue
                if any(not np.all(np.isfinite(unit[m][i])) for m in SV_MODELS):
                    continue
                self.vec[(r["group"], r["audio_path"])] = {m: unit[m][i] for m in SV_MODELS}

    def real_paths(self, excluded: set[str] | None = None) -> list[str]:
        ex = excluded or set()
        return sorted(p for (g, p) in self.vec if g == "real" and p not in ex)

    def get(self, group: str, path: str, model: str):
        v = self.vec.get((group, path))
        return None if v is None else v[model]


def pool_centroid(bank: EmbStore, model: str, excluded: set[str]) -> tuple[np.ndarray, list[str]]:
    pool = bank.real_paths(excluded)
    if len(pool) < 3:
        raise ValueError(f"real pool has {len(pool)} clips; is REAL_BANK_DIR={REAL_BANK_DIR} right?")
    return np.mean([bank.get("real", p, model) for p in pool], axis=0), pool


def voice_scores(store: EmbStore, bank: EmbStore, synth_paths: list[str], excluded: set[str]) -> dict[str, dict[str, float]]:
    """model -> {synth audio_path -> cosine to the real-pool centroid}; clips without an embedding are absent."""
    out: dict[str, dict[str, float]] = {}
    for m in SV_MODELS:
        c, _ = pool_centroid(bank, m, excluded)
        res = {}
        for p in synth_paths:
            e = store.get("synthetic", p, m)
            if e is not None:
                res[p] = float(e @ c / (np.linalg.norm(c) + 1e-10))
        out[m] = res
    return out


def real_ceiling(bank: EmbStore, real_paths: list[str], excluded: set[str]) -> dict[str, dict]:
    """Leave-one-out cosine of real clips to the rest of the pool (nothing synthetic is expected to beat it)."""
    out = {}
    for m in SV_MODELS:
        c, pool = pool_centroid(bank, m, excluded)
        n, in_pool = len(pool), set(pool)
        vals = []
        for p in real_paths:
            if p in in_pool:
                e = bank.get("real", p, m)
                vals.append(_cos(e, (c * n - e) / (n - 1)))
        out[m] = {"mean": mean_or_nan(vals), "n": len(vals)}
    return out


# ----------------------------------------------------------------------------- PathBench
def load_pb_scored(paths: list[str]) -> list[dict]:
    """Union of items_scored.csv files keyed by (severity, audio_path); a metric that is blank/NaN in one file
    is filled from another (e.g. a later --dartp pass), so a DArtP rerun can be merged with the cheap pass."""
    merged: dict[tuple[str, str], dict] = {}
    for p in paths:
        for r in read_csv(p):
            k = (r["severity"], r["audio_path"])
            cur = merged.setdefault(k, dict(r))
            for m in PB_METRICS:
                if m in r and np.isfinite(fl(r[m])) and not np.isfinite(fl(cur.get(m))):
                    cur[m] = r[m]
    return list(merged.values())


def pb_files(*dirs: str) -> list[str]:
    """items_scored.csv files under the given dirs (dir itself or one level of chunk sub-dirs)."""
    out = []
    for d in dirs:
        out += sorted(glob.glob(os.path.join(d, "items_scored.csv")) + glob.glob(os.path.join(d, "*", "items_scored.csv")))
    return out


def pb_sentence_means(rows: list[dict], group: str, metric: str, norms: set[str] | None = None) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for r in rows:
        if r["severity"] != group:
            continue
        n = norm(r["transcript"])
        v = fl(r.get(metric))
        if (norms is None or n in norms) and np.isfinite(v):
            acc.setdefault(n, []).append(v)
    return {n: float(np.mean(v)) for n, v in acc.items()}


def pb_record(rows: list[dict], group: str, targets: dict[str, dict], metrics=("per", "artp", "dartp", "dper")) -> tuple[dict, dict]:
    """(flat dict {metric, metric_n, ...}, per-sentence {metric: {norm: value}})."""
    out, sent = {}, {}
    names = set(targets)
    for m in metrics:
        d = pb_sentence_means(rows, group, m, names)
        sent[m] = d
        out[m] = mean_or_nan(d.values())
        out[f"{m}_n"] = len(d)
    return out, sent


def paired_se(a: dict[str, float], b: dict[str, float]) -> dict:
    """Per-sentence difference a - b over the sentences both have (finite): mean, SE (sd/sqrt n), n."""
    ks = [k for k in sorted(set(a) & set(b)) if np.isfinite(fl(a[k])) and np.isfinite(fl(b[k]))]
    d = np.array([fl(a[k]) - fl(b[k]) for k in ks], float)
    if d.size < 2:
        return {"mean": float(d.mean()) if d.size else float("nan"), "se": float("nan"), "n": int(d.size)}
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(d.size)), "n": int(d.size)}


# ----------------------------------------------------------------------------- one-run records
def score_run(label: str, run_name: str, targets: dict[str, dict], pb_rows: list[dict], store: EmbStore,
              bank: EmbStore, pb_group: str | None = None, gen_root: str | None = None) -> dict:
    """All headline metrics of one generated run on the target sentences. `pb_group` is the run's `severity`
    label in the scored PathBench csv (default: the run name)."""
    synth, n_rows = load_synth(run_name, gen_root)
    ratios = duration_ratios(synth, targets)
    rec: dict = {"label": label, "run": run_name, "n_targets": len(targets),
                 "n_synth": sum(1 for n in targets if n in synth), "n_rows": n_rows}
    ds = duration_stats(list(ratios.values()))
    rec.update({f"dur_{k}": v for k, v in ds.items()})
    pbr, sent = pb_record(pb_rows, pb_group or run_name, targets)
    rec.update(pbr)
    paths = [synth[n]["audio_path"] for n in targets if n in synth]
    excl = references_excluded(run_name, gen_root)
    vs = voice_scores(store, bank, paths, excl)
    for m in SV_MODELS:
        rec[m] = mean_or_nan(vs[m].values())
        rec[f"{m}_n"] = len(vs[m])
    rec["n_pool"] = len(bank.real_paths(excl))
    sent["dur_ratio"] = ratios
    rec["_sent"] = sent
    return rec


def score_real(targets: dict[str, dict], pb_rows: list[dict], bank: EmbStore, excluded: set[str],
               group: str = "real") -> dict:
    pbr, sent = pb_record(pb_rows, group, targets)
    rec: dict = {"label": "real", "n_targets": len(targets), "n_clips": sum(t["n_real"] for t in targets.values())}
    rec.update(pbr)
    paths = [p for t in targets.values() for p in t["audio_paths"]]
    ceil = real_ceiling(bank, paths, excluded)
    for m in SV_MODELS:
        rec[m] = ceil[m]["mean"]
        rec[f"{m}_n"] = ceil[m]["n"]
    rec["_sent"] = sent
    return rec


def mean_records(recs: list[dict]) -> dict:
    """Seed-average of numeric fields of several score_run records (same label); keeps per-seed values."""
    out = {k: v for k, v in recs[0].items() if not isinstance(v, (int, float)) or isinstance(v, bool)}
    keys = [k for k, v in recs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for k in keys:
        out[k] = mean_or_nan([r[k] for r in recs])
    out["n_seeds"] = len(recs)
    return out


# ----------------------------------------------------------------------------- tiny CLI = unit check on any existing run
def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Print ft_metrics' record for an existing generated run (e.g. a zero-shot "
                                             "baseline) on a split's sentences; used to unit-test the helpers.")
    ap.add_argument("--run", default="R1b_stitch10_s1")
    ap.add_argument("--split", default="tts_val", choices=["tts_val", "tts_test"])
    ap.add_argument("--pb", nargs="*", default=[f"{PB_EVAL}/out/patient_gen/items_scored.csv"],
                    help="scored PathBench csv(s); group label is the run name")
    ap.add_argument("--real-group", default="real")
    args = ap.parse_args()

    splits = load_splits()
    targets = split_targets(splits, args.split)
    pb = load_pb_scored(args.pb)
    bank = EmbStore([REAL_BANK_DIR])
    store = EmbStore([f"{OUT_V2}/gen_{args.run}"])
    rec = score_run(args.run, args.run, targets, pb, store, bank)
    real = score_real(targets, pb, bank, references_excluded(args.run))
    print(f"{args.split}: {len(targets)} sentences, {real['n_clips']} real clips")
    for name, r in (("real", real), (args.run, rec)):
        print(name, json.dumps(strip_private(r), indent=1))


if __name__ == "__main__":
    _cli()
