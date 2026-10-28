"""One Higgs generation plan for the item-3 stack: each patient's own adapter (B_r64 epoch 8) + its own ASR-train-only
ft_stitch5 reference, speaking the H pool A + pool B sentences (asr_aug/data/pool_{A,B}.json, the ONS text), minus any
sentence equal to a val / test / oov sentence of either patient. Seed 1, each sentence once.

    python make_gen_plan.py   # -> /scratch/pa2753/stack/data/plan_gen.json; run folders gen_runs/FT_ST_<p>_ft_stitch5_text_<p>_s1
"""
import json
import os
import re
import subprocess
import sys

HIGGS_FT = "/scratch/pa2753/higgs_ft"
S = "/scratch/pa2753/stack"
POOLS = "/scratch/pa2753/asr_aug/data"
ROLES = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
PATIENTS = ["s1", "s2"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower().replace("-", " "))).strip()


def main() -> None:
    held = {norm(json.loads(l)["transcript"]) for p in PATIENTS for r in ("val", "test", "oov")
            for l in open(f"{ROLES}/{p}_{r}.jsonl")}
    sents = json.load(open(f"{POOLS}/pool_A.json")) + json.load(open(f"{POOLS}/pool_B.json"))
    keep = [s for s in sents if norm(s["norm"]) not in held]
    print(f"pool sentences {len(sents)}, dropped as val/test/oov text {len(sents) - len(keep)}")
    os.makedirs(f"{S}/data", exist_ok=True)
    json.dump(keep, open(f"{S}/data/text.json", "w"), indent=1)
    units, model = [], None
    for p in PATIENTS:
        out = f"{S}/data/plan_{p}.json"
        subprocess.run([sys.executable, f"{HIGGS_FT}/build_plan_ft.py", "--out", out, "--sentences-file", f"{S}/data/text.json",
                        "--ref", "ft_stitch5", "--ref-json", f"{S}/higgs/{p}/refs/refs_ft.json", "--seeds", "1",
                        "--adapter", f"ST_{p}={S}/higgs/{p}/runs/B_r64/epoch_08"], check=True, cwd=HIGGS_FT,
                       env={**os.environ, "HIGGS_FT_ROOT": f"{S}/higgs/{p}"})
        pl = json.load(open(out))
        units += pl["units"]
        model = pl["model_path"]
    json.dump({"model_path": model, "units": units}, open(f"{S}/data/plan_gen.json", "w"), indent=1)
    print(f"{len(units)} units, {sum(len(u['prompts']) for u in units)} clips")


if __name__ == "__main__":
    main()
