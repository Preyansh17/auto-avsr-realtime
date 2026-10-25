"""Experiment RS: how much does the inference reference matter? (README, "Follow-up experiments", experiment R)

    python ref_study.py refs     # build candidate references from fold-0 TRAINING clips -> refs/refs_study.json
    python ref_study.py items    # PathBench items + neural-job list for the new runs -> exp_R/
    python ref_study.py table    # per-reference table, paired comparison vs ft_stitch5, sensitivity of B vs zero-shot

Design: fold 0 of the 5-fold cross-fit (46 held-out sentences, 3 seeds). Adapter B_r64_cv0 (epoch 8) and the pretrained model
(adapter 'none') are run with each reference, so the spread across references is measured for the finetuned and the
zero-shot model. Every reference is built only from clips of fold 0's training set (so no held-out sentence is ever in a
prompt); the ft_stitch5 rows are the existing experiment-K runs. Reference draw noise is read from rand5a vs rand5b and from
the seed sd. Nothing here selects a reference on test results alone: a winner has to be confirmed on a second fold.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_metrics as M  # noqa: E402

FOLD = 0
SEEDS = [1, 2, 3]
ADAPTERS = {"B": "B_r64_cv0", "ZS": "pretrained_cv0"}      # label -> adapter tag used in run names
NEW_REFS = ["stitch5", "lowpitch5", "rs_rand1", "rs_rand3", "rs_rand5a", "rs_rand5b", "rs_vocab5"]
ALL_REFS = ["ft_stitch5"] + NEW_REFS
REF_SEED = 20261009
OUT = f"{M.ROOT}/exp_RS"
NBOOT = 10000


def run_name(adapter: str, ref: str, seed: int) -> str:
    return f"FT_{adapter}_{ref}_tts_test_s{seed}"


def cv_splits() -> dict:
    return json.load(open(f"{M.ROOT}/data/cv/splits_cv{FOLD}.json"))


# ----------------------------------------------------------------------------- refs
def cmd_refs(a) -> None:
    sys.path.insert(0, os.environ.get("HIGGS_GEN", "/scratch/pa2753/higgs_gen"))
    from make_refs import stitch

    sp = cv_splits()
    clips, train = sp["clips"], sp["tts_train"]
    test_norms = {clips[u]["norm"] for u in sp["tts_test"]}
    train_paths = {clips[u]["audio_path"] for u in train}
    pool = [u for u in train if 4.0 <= clips[u]["duration_sec"] <= 12.0]
    assert not ({clips[u]["norm"] for u in pool} & test_norms)
    out_dir = f"{M.ROOT}/refs/study"
    os.makedirs(out_dir, exist_ok=True)
    existing = json.load(open(f"{M.ROOT}/refs/refs_ft.json"))["refs"]
    refs: dict = {}
    used_norms: set[str] = {clips[u]["norm"] for u in existing["ft_stitch5"]["source_utt_ids"]}

    def build(key: str, picked: list[str], how: str) -> None:
        o = f"{out_dir}/{key}.wav"
        st = stitch([clips[u]["audio_path"] for u in picked], o)
        assert st["duration_sec"] <= 46.0, (key, st["duration_sec"])
        refs[key] = {"audio_path": o, "transcript": " ".join(clips[u]["transcript"] for u in picked),
                     "source_utt_ids": picked, "source_audio_paths": [clips[u]["audio_path"] for u in picked],
                     "duration_sec": st["duration_sec"], "origin": how}
        used_norms.update(clips[u]["norm"] for u in picked)

    def pick(rng, n, cap=45.0, lo=0.0, avoid=()):
        cands = [u for u in pool if clips[u]["norm"] not in avoid and clips[u]["duration_sec"] >= lo]
        rng.shuffle(cands)
        got, norms, tot = [], set(), 0.0
        for u in cands:
            c = clips[u]
            if c["norm"] in norms or tot + c["duration_sec"] > cap:
                continue
            got.append(u); norms.add(c["norm"]); tot += c["duration_sec"]
            if len(got) == n:
                break
        assert len(got) == n, (n, len(got))
        return got

    rng = random.Random(REF_SEED)
    build("rs_rand1", pick(rng, 1, lo=6.0, avoid=used_norms), "1 random clip >= 6 s")
    build("rs_rand3", pick(rng, 3, avoid=used_norms), "3 random clips")
    build("rs_rand5a", pick(rng, 5, avoid=used_norms), "5 random clips, draw a (disjoint sentences from ft_stitch5)")
    build("rs_rand5b", pick(rng, 5, avoid=used_norms), "5 random clips, draw b (disjoint from a)")
    # vocabulary-diverse: greedy, most new words per clip, total <= 45 s, distinct sentences
    words: set[str] = set()
    got, tot, norms = [], 0.0, set()
    cands = list(pool)
    random.Random(REF_SEED + 1).shuffle(cands)
    while len(got) < 5:
        best = max((u for u in cands if clips[u]["norm"] not in norms and clips[u]["norm"] not in used_norms
                    and tot + clips[u]["duration_sec"] <= 45.0 - 4.0 * (4 - len(got))),
                   key=lambda u: len(set(clips[u]["norm"].split()) - words), default=None)
        assert best is not None
        got.append(best); words |= set(clips[best]["norm"].split()); norms.add(clips[best]["norm"]); tot += clips[best]["duration_sec"]
    build("rs_vocab5", got, "greedy: 5 clips maximising distinct words")
    for k in ("stitch5", "lowpitch5"):
        v = existing[k]
        miss = [p for p in v["source_audio_paths"] if p not in train_paths]
        assert not miss, f"{k}: sources not in fold-0 training set: {miss}"
        refs[k] = {x: v[x] for x in ("audio_path", "transcript", "source_audio_paths", "duration_sec")}
        refs[k]["origin"] = "existing zero-shot reference, all sources in fold-0 training set"
    refs["ft_stitch5"] = existing["ft_stitch5"]
    for k, v in refs.items():
        nw = len({w for t in [v["transcript"].lower().split()] for w in t})
        print(f"{k:11s} {v['duration_sec']:5.1f}s {len(v['source_audio_paths'])} clips {nw:3d} distinct words  [{v['origin']}]")
    json.dump({"refs": refs, "fold": FOLD, "split_hash": sp["log"]["split_hash"]}, open(f"{M.ROOT}/refs/refs_study.json", "w"), indent=1)


# ----------------------------------------------------------------------------- items
def runs_new() -> list[str]:
    # stitch5 / lowpitch5 runs under these names were generated and scored by experiment R (exp_R); only rs_* are new here
    return [run_name(ad, r, s) for ad in ADAPTERS.values() for r in NEW_REFS if r.startswith("rs_") for s in SEEDS]


def runs_all() -> list[str]:
    return [run_name(ad, r, s) for ad in ADAPTERS.values() for r in ALL_REFS for s in SEEDS]


def cmd_items(a) -> None:
    sp = cv_splits()
    targets = M.split_targets(sp, "tts_test")
    os.makedirs(OUT, exist_ok=True)
    # real clips and the ft_stitch5 runs are scored already in exp_K; only the new runs go to PathBench here
    items, problems = [], []
    for r in runs_new():
        c = M.run_completeness(r, targets)
        if not c["ok"]:
            problems.append(f"{r}: {c['n_present']}/{c['n_targets']} missing audio {len(c['missing_audio'])}")
            continue
        items += M.synth_items(r, r, targets)
    if problems and not a.allow_incomplete:
        sys.exit("incomplete:\n  " + "\n  ".join(problems))
    M.write_csv(f"{OUT}/pb_items_all.csv", items, M.ITEM_FIELDS)
    for i in range(a.chunks):
        M.write_csv(f"{OUT}/pb_items_{i:02d}.csv", items[i :: a.chunks], M.ITEM_FIELDS)
    todo = [f"ms_{r}|{r}" for r in runs_all()
            if not os.path.isdir(f"{M.OUT_V2}/ms_{r}") or not any(f.startswith("neural.shard") for f in os.listdir(f"{M.OUT_V2}/ms_{r}"))]
    open(f"{OUT}/neural_todo.txt", "w").write("\n".join(todo) + ("\n" if todo else ""))
    print(f"{len(items)} PathBench items in {a.chunks} chunks; {len(todo)} runs need neural embeddings")


# ----------------------------------------------------------------------------- table
def boot(d: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(d), (NBOOT, len(d)))
    v = d[idx].mean(1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def cmd_table(a) -> None:
    sp = cv_splits()
    targets = M.split_targets(sp, "tts_test")
    refs = json.load(open(f"{M.ROOT}/refs/refs_study.json"))["refs"]
    pb = M.load_pb_scored(M.pb_files(f"{M.ROOT}/exp_K/pb_out", f"{M.ROOT}/exp_R/pb_out", f"{OUT}/pb_out"))
    dirs = sorted({f"{M.OUT_V2}/ms_{r}" for r in runs_all()})
    for x in dirs + [M.REAL_BANK_DIR]:
        if not os.path.isdir(x):
            sys.exit(f"{x} missing (neural scoring not finished?)")
    store, bank = M.EmbStore(dirs), M.EmbStore([M.REAL_BANK_DIR])
    real = M.score_real(targets, pb, bank, M.references_excluded(run_name(ADAPTERS["B"], "ft_stitch5", 1)))
    R = real["_sent"]
    rec: dict[tuple[str, str], list[dict]] = {}
    for lab, ad in ADAPTERS.items():
        for ref in ALL_REFS:
            rec[(lab, ref)] = [M.score_run(run_name(ad, ref, s), run_name(ad, ref, s), targets, pb, store, bank) for s in SEEDS]
            bad = [r["label"] for r in rec[(lab, ref)] if r["n_synth"] < r["n_targets"]]
            if bad:
                print("WARNING incomplete", bad, flush=True)

    def smean(key, m):
        rs = [r["_sent"][m] for r in rec[key]]
        ks = set.intersection(*[set(x) for x in rs])
        return {k: float(np.mean([x[k] for x in rs])) for k in ks}

    keys = set(R["per"]) & set(R["artp"])
    for key in rec:
        keys &= set(smean(key, "per")) & set(smean(key, "artp"))
    keys = sorted(keys)
    err = {key: np.array([smean(key, "per")[k] - R["per"][k] for k in keys]) for key in rec}
    aerr = {key: np.array([smean(key, "artp")[k] - R["artp"][k] for k in keys]) for key in rec}

    def agg(key):
        rs = rec[key]
        o = {}
        for m in ("per", "artp", "titanet", "wavlm", "dur_within20", "dur_ramble", "dur_median"):
            v = np.array([r[m] for r in rs], float)
            o[m], o[m + "_sd"] = float(np.nanmean(v)), float(np.nanstd(v, ddof=1))
        return o

    A = {key: agg(key) for key in rec}
    rng = np.random.default_rng(0)
    md = [f"# Experiment RS: reference study (fold {FOLD}, {len(targets)} sentences, {len(keys)} with PER and ArtP in every row; seeds {SEEDS})", "",
          f"Real PER {real['per']:.3f}, ArtP {real['artp']:.3f}, TitaNet ceiling {real['titanet']:.3f}. `bias` = mean per-sentence (synthetic - real), "
          "`MAE` = mean absolute per-sentence PER error, `d MAE` = MAE minus the ft_stitch5 row of the same model (paired bootstrap 95% CI over sentences; "
          "negative = better than ft_stitch5).", "", "## References", "",
          "| reference | seconds | clips | distinct words | origin |", "|---|---|---|---|---|"]
    for r in ALL_REFS:
        v = refs[r]
        md.append(f"| {r} | {v['duration_sec']:.1f} | {len(v['source_audio_paths'])} | {len({w for w in v['transcript'].lower().split()})} | {v.get('origin', '')} |")
    out_json: dict = {"fold": FOLD, "n_sentences": len(keys), "rows": {}}
    for lab, title in (("B", "Finetuned B_r64 (fold 0, epoch 8)"), ("ZS", "Zero-shot (pretrained, same references)")):
        md += ["", f"## {title}", "",
               "| reference | PER | PER bias | PER MAE | d MAE vs ft_stitch5 [95% CI] | ArtP | ArtP bias | TitaNet | within +-20% | ramble |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        base = err[(lab, "ft_stitch5")]
        for r in ALL_REFS:
            key, x = (lab, r), A[(lab, r)]
            dm = np.abs(err[key]) - np.abs(base)
            ci = boot(dm, rng) if r != "ft_stitch5" else (float("nan"),) * 2
            row = {"per": x["per"], "per_sd": x["per_sd"], "bias": float(err[key].mean()), "mae": float(np.abs(err[key]).mean()),
                   "d_mae": float(dm.mean()), "d_mae_ci": list(ci), "artp": x["artp"], "artp_bias": float(aerr[key].mean()),
                   "titanet": x["titanet"], "within20": x["dur_within20"], "ramble": x["dur_ramble"]}
            out_json["rows"][f"{lab}/{r}"] = row
            dtxt = "-" if r == "ft_stitch5" else f"{dm.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
            md.append(f"| {r} | {x['per']:.3f} +- {x['per_sd']:.3f} | {err[key].mean():+.3f} | {np.abs(err[key]).mean():.3f} | {dtxt} | "
                      f"{x['artp']:.3f} | {aerr[key].mean():+.3f} | {x['titanet']:.3f} | {100 * x['dur_within20']:.0f}% | {100 * x['dur_ramble']:.1f}% |")
    md += ["", "## Sensitivity to the reference (spread across the 8 references)", "",
           "| model | PER bias range | PER bias sd | MAE range | TitaNet range | within +-20% range | ramble range | mean seed sd of PER |", "|---|---|---|---|---|---|---|---|"]
    sens = {}
    for lab, nm in (("B", "finetuned B"), ("ZS", "zero-shot")):
        b = np.array([err[(lab, r)].mean() for r in ALL_REFS]); mae = np.array([np.abs(err[(lab, r)]).mean() for r in ALL_REFS])
        t = np.array([A[(lab, r)]["titanet"] for r in ALL_REFS]); w = np.array([A[(lab, r)]["dur_within20"] for r in ALL_REFS])
        rb = np.array([A[(lab, r)]["dur_ramble"] for r in ALL_REFS]); sd = np.mean([A[(lab, r)]["per_sd"] for r in ALL_REFS])
        sens[lab] = {"bias_range": float(np.ptp(b)), "bias_sd": float(b.std(ddof=1)), "mae_range": float(np.ptp(mae)), "titanet_range": float(np.ptp(t)),
                     "within_range": float(np.ptp(w)), "ramble_range": float(np.ptp(rb)), "seed_sd": float(sd)}
        md.append(f"| {nm} | {np.ptp(b):.3f} | {b.std(ddof=1):.3f} | {np.ptp(mae):.3f} | {np.ptp(t):.3f} | {100 * np.ptp(w):.0f} pts | {100 * np.ptp(rb):.1f} pts | {sd:.3f} |")
    out_json["sensitivity"] = sens
    d5 = abs(err[("B", "rs_rand5a")].mean() - err[("B", "rs_rand5b")].mean())
    md += ["", f"Reference-draw noise: finetuned B, rand5a vs rand5b differ by {d5:.3f} in PER bias "
           f"(seed sd of PER {sens['B']['seed_sd']:.3f}).", ""]
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/table.md", "w").write("\n".join(md) + "\n")
    json.dump(out_json, open(f"{OUT}/table.json", "w"), indent=1)
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refs")
    i = sub.add_parser("items"); i.add_argument("--chunks", type=int, default=3); i.add_argument("--allow-incomplete", action="store_true")
    sub.add_parser("table")
    a = ap.parse_args()
    {"refs": cmd_refs, "items": cmd_items, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
