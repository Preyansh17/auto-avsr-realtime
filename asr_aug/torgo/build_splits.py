"""TORGO dysarthric speakers -> pooled ASR splits for the Higgs augmentation extension (step 6, approved 2026-10-15).

    python build_splits.py [--out /scratch/pa2753/torgo_aug/data]

Items: every head-mic wav of the 8 dysarthric speakers with a readable prompt. Dropped: non-verbal instructions
("[say Ah-P-Eee repeatedly]"), picture-description prompts (*.jpg, unknown transcript), empty text.
Split by normalised prompt TEXT, not by clip: each unique text goes to train / val / test for every speaker, so no test
or val word or sentence is ever heard in training (TORGO speakers read a shared prompt list). 60/20/20 over unique texts,
stratified by type (word = 1 token, sentence = 2+). Writes items.csv (all items with split), split_summary.md, audit.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import re

import soundfile as sf

RAW = "/scratch/th3482/TORGO/raw"
SPEAKERS = {"F01": "severe", "F03": "mild", "F04": "mild", "M01": "severe", "M02": "severe", "M03": "mild",
            "M04": "severe", "M05": "mild"}
SEED = 20261015
FRAC = {"train": 0.6, "val": 0.2, "test": 0.2}


def norm(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    return " ".join(w.strip("'") for w in t.split() if w.strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/pa2753/torgo_aug/data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items, dropped = [], collections.Counter()
    for spk in SPEAKERS:
        for wav in sorted(glob.glob(f"{RAW}/{spk}/Session*/wav_headMic/*.wav")):
            sess = wav.split("/")[-3]
            pf = f"{RAW}/{spk}/{sess}/prompts/{os.path.basename(wav)[:-4]}.txt"
            if not os.path.exists(pf):
                dropped["no_prompt"] += 1
                continue
            raw = open(pf, errors="replace").read().strip()
            if "[" in raw or ".jpg" in raw.lower() or "/" in raw:
                dropped["instruction_or_picture"] += 1
                continue
            text = norm(raw)
            if not text:
                dropped["empty"] += 1
                continue
            try:
                info = sf.info(wav)
            except RuntimeError:
                dropped["unreadable"] += 1
                continue
            if info.duration < 0.2:
                dropped["too_short"] += 1
                continue
            items.append({"speaker": spk, "severity": SPEAKERS[spk], "session": sess, "wav": wav, "prompt_raw": raw,
                          "text": text, "type": "word" if len(text.split()) == 1 else "sentence",
                          "duration": round(info.duration, 3), "sr": info.samplerate})
    rng = random.Random(SEED)
    split_of = {}
    for typ in ("word", "sentence"):
        texts = sorted({i["text"] for i in items if i["type"] == typ})
        rng.shuffle(texts)
        n = len(texts)
        cut1, cut2 = round(n * FRAC["train"]), round(n * (FRAC["train"] + FRAC["val"]))
        for k, t in enumerate(texts):
            split_of[t] = "train" if k < cut1 else "val" if k < cut2 else "test"
    for i in items:
        i["split"] = split_of[i["text"]]
    with open(f"{a.out}/items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]))
        w.writeheader()
        w.writerows(items)
    # audit: no text shared across splits; vocabulary of val/test words unseen in train
    by_split = collections.defaultdict(set)
    for i in items:
        by_split[i["split"]].add(i["text"])
    train_vocab = {w for t in by_split["train"] for w in t.split()}
    audit = {"seed": SEED, "dropped": dict(dropped), "n_items": len(items),
             "text_overlap": {f"{x}&{y}": len(by_split[x] & by_split[y]) for x, y in
                              (("train", "val"), ("train", "test"), ("val", "test"))},
             "unique_texts": {s: len(v) for s, v in by_split.items()},
             "oov_token_rate": {s: round(sum(w not in train_vocab for t in by_split[s] for w in t.split()) /
                                         max(1, sum(len(t.split()) for t in by_split[s])), 3) for s in ("val", "test")}}
    assert all(v == 0 for v in audit["text_overlap"].values()), audit["text_overlap"]
    json.dump(audit, open(f"{a.out}/audit.json", "w"), indent=1)
    L = ["# TORGO pooled split", "", f"Dropped: {dict(dropped)}. Kept {len(items)} clips. Unique texts: "
         f"{audit['unique_texts']}. OOV token rate vs train: {audit['oov_token_rate']}.", "",
         "| speaker | severity | train words | train sents | val words | val sents | test words | test sents | train min |",
         "|---|---|---|---|---|---|---|---|---|"]
    for spk, sev in SPEAKERS.items():
        c = collections.Counter((i["split"], i["type"]) for i in items if i["speaker"] == spk)
        mins = sum(i["duration"] for i in items if i["speaker"] == spk and i["split"] == "train") / 60
        L.append(f"| {spk} | {sev} | " + " | ".join(str(c[(s, t)]) for s in ("train", "val", "test")
                                                    for t in ("word", "sentence")) + f" | {mins:.1f} |")
    c = collections.Counter((i["split"], i["type"]) for i in items)
    L.append("| all | | " + " | ".join(str(c[(s, t)]) for s in ("train", "val", "test") for t in ("word", "sentence"))
             + f" | {sum(i['duration'] for i in items if i['split'] == 'train') / 60:.1f} |")
    open(f"{a.out}/split_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
