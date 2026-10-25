"""One Higgs generation plan for all 8 TORGO speakers: each speaker's own adapter (B_r64 epoch 8) + its own ASR-train-only
ft_stitch5 reference, speaking its text_<spk>.json (160 words + 60 sentences), seed 1.

    python make_gen_plan.py   # -> /scratch/pa2753/torgo_aug/data/plan_gen.json; run folders gen_runs/FT_TG_<spk>_ft_stitch5_text_<spk>_s1
"""
import json
import os
import subprocess
import sys

HIGGS_FT = "/scratch/pa2753/higgs_ft"
T = "/scratch/pa2753/torgo_aug"
SPEAKERS = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]


def main() -> None:
    units, model = [], None
    for spk in SPEAKERS:
        adapter = f"{T}/higgs/{spk}/runs/B_r64/epoch_08"
        out = f"{T}/data/plan_{spk}.json"
        subprocess.run([sys.executable, f"{HIGGS_FT}/build_plan_ft.py", "--out", out, "--sentences-file", f"{T}/data/text_{spk}.json",
                        "--ref", "ft_stitch5", "--ref-json", f"{T}/higgs/{spk}/refs/refs_ft.json", "--seeds", "1",
                        "--adapter", f"TG_{spk}={adapter}"], check=True, cwd=HIGGS_FT,
                       env={**os.environ, "HIGGS_FT_ROOT": f"{T}/higgs/{spk}"})
        p = json.load(open(out))
        units += p["units"]
        model = p["model_path"]
    json.dump({"model_path": model, "units": units}, open(f"{T}/data/plan_gen.json", "w"), indent=1)
    print(f"{len(units)} units, {sum(len(u['prompts']) for u in units)} clips")


if __name__ == "__main__":
    main()
