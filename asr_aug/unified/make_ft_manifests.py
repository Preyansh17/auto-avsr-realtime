"""NeMo manifests for the unified-model fine-tune (PLAN.md 3.3 item 0, step 2), from the shared role files (read only).

    python make_ft_manifests.py --out /scratch/pa2753/unified/ft/manifests [--limit 20] [--recipe e2|v2r]

e2 (default): train = old_legacy_train + old_legal_train + s1_train + s2_train (314 + 240 + 240 = 794 real clips, the
E2 real data); val = s1_val + s2_val (60).
v2r (semg2 v2, real only): train = s2ov_train + s2_train (189 + 240 = 429; s2ov_* = Tianyu's 2026-10-16 open-vocab
split, checked identical to his package); val = s2ov_val + s2_val (55).
v2h: v2r + the 5,378 synthetic rows of Tianyu's phone model (his package train.csv, three Higgs pools, already
filtered by him against val25/test-open) = 5,807 train rows; same val.
v2o: v2r + our own 1,751 kept semg2 Higgs clips (stack study: semg2 adapter trained on s2_train only, ft_stitch5
reference, pool A/B text, H filter; `stack/data/filter_clips.csv` kept rows) = 2,180 train rows; same val. Their text
has no exact or >= 3-word contained match in val25, test-open38, conv34, test30 or oov50 (checked 2026-10-29).
Each row carries text and source_lang/lang = "en" (the unified model's lang_field).
"""
import argparse
import csv
import json
import os

import soundfile as sf

ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
RECIPES = {"e2": {"train": ["old_legacy_train", "old_legal_train", "s1_train", "s2_train"], "val": ["s1_val", "s2_val"]},
           "v2r": {"train": ["s2ov_train", "s2_train"], "val": ["s2ov_val", "s2_val"]}}
RECIPES["v2h"] = RECIPES["v2o"] = RECIPES["v2r"]
OURS = "/scratch/pa2753/stack/data"; OURS_AUDIO = "/scratch/pa2753/gen_runs"
TIANYU_TRAIN = "/scratch/pa2753/tianyu_semg2_best_v1/semg2_best_v1_20261029/phone_streaming_nemotron/data/train.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="rows per role (smoke test)")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="e2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, roles in RECIPES[a.recipe].items():
        n = 0
        with open(f"{a.out}/{name}.json", "w") as f:
            for role in roles:
                rows = [json.loads(l) for l in open(f"{ROLES}/{role}.jsonl")]
                for r in rows[: a.limit or None]:
                    f.write(json.dumps({"audio_filepath": r["audio_path"], "duration": sf.info(r["audio_path"]).duration,
                                        "text": r["transcript"], "source_lang": "en", "lang": "en",
                                        "take_id": r.get("take_id") or r["rel_path"], "role": role}) + "\n")
                    n += 1
            if a.recipe == "v2o" and name == "train":
                text = {t["norm"]: t["text"] for t in json.load(open(f"{OURS}/text.json"))}  # cased form, as in real rows
                syn = [r for r in csv.DictReader(open(f"{OURS}/filter_clips.csv")) if r["patient"] == "s2" and r["kept"] == "True"]
                for r in syn[: a.limit or None]:
                    p = f"{OURS_AUDIO}/{r['rel_path']}"
                    f.write(json.dumps({"audio_filepath": p, "duration": sf.info(p).duration, "text": text[r["norm"]],
                                        "source_lang": "en", "lang": "en", "take_id": r["prompt_id"],
                                        "role": f"ours_s2_{r['pool']}"}) + "\n")
                    n += 1
            if a.recipe == "v2h" and name == "train":
                syn = [r for r in csv.DictReader(open(TIANYU_TRAIN)) if r["dataset"].startswith("synth_")]
                for r in syn[: a.limit or None]:
                    f.write(json.dumps({"audio_filepath": r["torch_path"], "duration": sf.info(r["torch_path"]).duration,
                                        "text": r["text"], "source_lang": "en", "lang": "en",
                                        "take_id": r["file"], "role": r["dataset"]}) + "\n")
                    n += 1
        print(name, n)


if __name__ == "__main__":
    main()
