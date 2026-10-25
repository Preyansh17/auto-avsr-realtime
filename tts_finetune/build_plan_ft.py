"""Lay out generation runs for finetuned adapters, in the format gen_higgs/gen_ft and the v2 evaluation read.

    python build_plan_ft.py --out plan_val.json --split tts_val --ref ft_stitch5 --seeds 1 \
        --adapter A_r16_e05=/scratch/pa2753/higgs_ft/runs/A_r16/epoch_05 --adapter none=none

One run folder per (adapter tag, seed): /scratch/pa2753/gen_runs/FT_<tag>_<ref>_<split>_s<seed>, with
manifests/real_manifest.csv restricted to the split's real clips and manifests/references.csv, like
higgs_gen/build_plan.py. Prompt ids are build_plan.py's (stable_id("gen", norm)), so every run is paired
by prompt id with the zero-shot baselines on identical text. Seeds are build_plan.py's SEED1 + (seed-1).

Sentence sources: --split tts_val|tts_test (from data/splits.json) or --sentences-file (a json list of
{"text", "norm"[, "prompt_id"]}, e.g. the out-of-vocabulary probe; no real manifest is written for these).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from higgs_ft_lib import MODEL, SYSTEM, normalize_text, stable_id  # noqa: E402

ROOT = os.environ.get("HIGGS_FT_ROOT", "/scratch/pa2753/higgs_ft")
GEN = "/scratch/pa2753/gen_runs"
TH_REAL = "/scratch/th3482/TORGO/runs/patient_higgs_whisper_20260702_072656/manifests/real_manifest.csv"
DEFAULT = {"temperature": 1.0, "top_k": 50, "top_p": 0.95, "max_new_tokens": 2048,
           "ras_win_len": 7, "ras_win_max_num_repeat": 2}
SEED1 = 260619825
CHUNK = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["tts_val", "tts_test"])
    ap.add_argument("--sentences-file")
    ap.add_argument("--ref", required=True, help="key in refs_ft.json (or --ref-json)")
    ap.add_argument("--ref-json", default=f"{ROOT}/refs/refs_ft.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--adapter", action="append", required=True, help="TAG=PATH (PATH may be 'none')")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--splits-file", default=f"{ROOT}/data/splits.json", help="splits json holding the --split key (cross-fitting: data/cv/splits_cv<k>.json)")
    args = ap.parse_args()
    if bool(args.split) == bool(args.sentences_file):
        ap.error("give exactly one of --split / --sentences-file")

    refs = json.load(open(args.ref_json))
    ref = (refs.get("refs") or refs)[args.ref]
    label = args.split or os.path.splitext(os.path.basename(args.sentences_file))[0]

    real_rows = []
    if args.split:
        sp = json.load(open(args.splits_file))
        ids = set(sp[args.split])
        pid_by_norm = {s["norm"]: s["prompt_id"] for s in json.load(open(f"{GEN}/sentences.json"))}
        seen, sentences = set(), []
        for u in sp[args.split]:
            c = sp["clips"][u]
            if c["norm"] in seen:
                continue
            seen.add(c["norm"])
            sentences.append({"prompt_id": pid_by_norm[c["norm"]], "text": c["transcript"], "norm": c["norm"]})
        real_rows = [r for r in csv.DictReader(open(TH_REAL)) if r["utt_id"] in ids]
        assert len(real_rows) == len(ids), f"{len(real_rows)} real rows for {len(ids)} clips"
    else:
        sentences = []
        for s in json.load(open(args.sentences_file)):
            norm = s.get("norm") or normalize_text(s["text"])
            sentences.append({"prompt_id": s.get("prompt_id") or stable_id("gen", norm, length=12),
                              "text": s["text"], "norm": norm})
    sentences.sort(key=lambda s: s["prompt_id"])

    tags = [a.split("=", 1)[0] for a in args.adapter]
    assert len(set(tags)) == len(tags), f"duplicate adapter tags: {tags}"
    for a in args.adapter:
        path = a.split("=", 1)[1]
        if path != "none":
            assert os.path.isfile(f"{path}/adapter_model.safetensors") and os.path.isfile(f"{path}/adapter_config.json"), \
                f"not a saved adapter: {path}"
    if args.split:  # an adapter must have been trained on the split file the sentences come from (cross-fitting guard)
        for a in args.adapter:
            path = a.split("=", 1)[1]
            if path != "none":
                cfg = json.load(open(os.path.join(os.path.dirname(path.rstrip("/")), "config.json")))
                assert cfg.get("split_hash") == sp["log"]["split_hash"], \
                    f"{path} was trained on split {cfg.get('split_hash')} but --splits-file is {sp['log']['split_hash']}"
    units = []
    for spec in args.adapter:
        tag, path = spec.split("=", 1)
        tag += args.tag_suffix
        for seed in args.seeds:
            name = f"FT_{tag}_{args.ref}_{label}_s{seed}"
            root = f"{GEN}/{name}"
            os.makedirs(f"{root}/manifests", exist_ok=True)
            rm = f"{root}/manifests/real_manifest.csv"
            if real_rows and os.path.exists(rm):
                old = {r["utt_id"] for r in csv.DictReader(open(rm))}
                assert old == {r["utt_id"] for r in real_rows}, f"{root} already holds a different sentence set; use a fold-specific --adapter tag"
            if real_rows:
                with open(f"{root}/manifests/real_manifest.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(real_rows[0].keys()))
                    w.writeheader()
                    w.writerows(real_rows)
            with open(f"{root}/manifests/references.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["utt_id", "speaker", "severity", "split", "transcript", "audio_path",
                                                  "duration_sec", "selection", "stitched_reference_count",
                                                  "stitched_source_audio_paths"])
                w.writeheader()
                w.writerow({"utt_id": f"ref_{args.ref}", "speaker": "bingge", "severity": "patient", "split": "train",
                            "transcript": ref["transcript"], "audio_path": ref["audio_path"],
                            "duration_sec": ref["duration_sec"], "selection": args.ref,
                            "stitched_reference_count": len(ref["source_audio_paths"]),
                            "stitched_source_audio_paths": "|".join(ref["source_audio_paths"])})
            for ci in range(0, len(sentences), CHUNK):
                units.append({
                    "name": f"{name}__c{ci // CHUNK}", "variant": name, "out_root": root, "speaker": "bingge",
                    "ref_wav": ref["audio_path"], "ref_text": ref["transcript"],
                    "ref_count": len(ref["source_audio_paths"]), "params": dict(DEFAULT), "system_prompt": SYSTEM,
                    "seed_base": SEED1 + seed - 1, "adapter": path,
                    "prompts": [{"prompt_id": s["prompt_id"], "text": s["text"]} for s in sentences[ci : ci + CHUNK]]})
    json.dump({"model_path": MODEL, "units": units}, open(args.out, "w"), indent=1)
    print(f"{len(units)} units, {sum(len(u['prompts']) for u in units)} clips -> {args.out}")


if __name__ == "__main__":
    main()
