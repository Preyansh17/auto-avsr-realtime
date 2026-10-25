"""Split the A/B sentence pools round-robin over reference prompts and build one Higgs generation plan.

    python make_gen_plan.py --data /scratch/pa2753/asr_aug/data --adapter /scratch/pa2753/higgs_ft/runs/B_r64/epoch_08

Each sentence is generated once, with one of REFS (experiment R: after finetuning the reference barely matters, so
rotating only adds diversity). Writes data/sents_<ref>.json, calls tts_finetune/build_plan_ft.py per ref (run folders
/scratch/pa2753/gen_runs/FT_AUG_B8_<ref>_sents_<ref>_s1) and merges the units into data/plan_gen.json.
"""
import argparse
import json
import os
import subprocess
import sys

REFS = ["ft_stitch5", "stitch5", "lowpitch5", "median1"]
HIGGS_FT = "/scratch/pa2753/higgs_ft"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()
    sents = json.load(open(f"{args.data}/pool_A.json")) + json.load(open(f"{args.data}/pool_B.json"))
    units = []
    for k, ref in enumerate(REFS):
        part = sents[k::len(REFS)]
        f = f"{args.data}/sents_{ref}.json"
        json.dump(part, open(f, "w"), indent=1)
        out = f"{args.data}/plan_{ref}.json"
        subprocess.run([sys.executable, f"{HIGGS_FT}/build_plan_ft.py", "--out", out, "--sentences-file", f,
                        "--ref", ref, "--seeds", "1", "--adapter", f"AUG_B8={args.adapter}"], check=True, cwd=HIGGS_FT)
        p = json.load(open(out))
        units += p["units"]
        model = p["model_path"]
    json.dump({"model_path": model, "units": units}, open(f"{args.data}/plan_gen.json", "w"), indent=1)
    print(f"{len(units)} units, {sum(len(u['prompts']) for u in units)} clips")


if __name__ == "__main__":
    main()
