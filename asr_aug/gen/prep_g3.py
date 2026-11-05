"""G3 training manifests (asr_aug/gen/README.md, amendment 4): VoxCPM2 LoRA per Higgs voice, on G1's clips.

    python3 prep_g3.py

Reads /scratch/pa2753/gen/g1/<v>/manifest.tsv (uid, text) and wavs/, writes /scratch/pa2753/gen/g3/<v>/train.jsonl
({"audio", "text"[, "ref_audio"]}) and steps.txt (3 epochs at effective batch 16, warmup 10% of the steps).
Text gets a capital first letter (VoxCPM wants sentence case). 40% of rows, seeded, get a ref_audio: another clip of
the same voice. Generation reuses g1/<v>/plan.json, so the plan rows are the ones the leak check already passed.
"""
import json
import math
import os
import random

G = "/scratch/pa2753/gen"
EPOCHS, EFF_BATCH, REF_SHARE = 3, 16, 0.4


def main():
    for i, v in enumerate(("old", "s1", "s2")):
        src, d = f"{G}/g1/{v}", f"{G}/g3/{v}"
        os.makedirs(d, exist_ok=True)
        rows = []
        for line in open(f"{src}/manifest.tsv"):
            uid, text = line.rstrip("\n").split("\t")
            wav = os.path.realpath(f"{src}/wavs/{uid}.wav")
            assert os.path.exists(wav), wav
            text = text.strip()
            rows.append({"audio": wav, "text": text[:1].upper() + text[1:]})
        rng = random.Random(20261104 + i)
        for r in rows:
            if rng.random() < REF_SHARE:
                r["ref_audio"] = rng.choice([x["audio"] for x in rows if x["audio"] != r["audio"]])
        with open(f"{d}/train.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        steps = math.ceil(EPOCHS * len(rows) / EFF_BATCH)
        warm = math.ceil(0.1 * steps)
        open(f"{d}/steps.txt", "w").write(f"{steps} {warm}\n")
        print(v, len(rows), "rows,", sum("ref_audio" in r for r in rows), "with ref_audio, steps", steps, "warmup", warm)


if __name__ == "__main__":
    main()
