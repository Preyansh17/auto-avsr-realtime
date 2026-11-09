#!/bin/bash
#SBATCH --job-name=sv_t2m_judge
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# A11/A12: zero-shot USR 2.0 Huge judge (step 1.2 settings) on the spliced sets (a, b, and c in step 1b) and on the real
# held-out crops (a_real; c_real in step 1b). T=t2m dir (default t2m; step 1b: t2m/step1b).
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" \
#   [--export=ALL,T=/scratch/pa2753/synvid/t2m/step1b] t2m_judge.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; T=${T:-$SV/t2m}
$SV/envs/usr2/bin/python - <<PY
import json, os
T = "$T"; SV = "$SV"; R = "/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles"
for name in ("a", "b", "c"):
    if not os.path.exists(f"{T}/{name}/rows.jsonl"):
        continue
    with open(f"{T}/judge_{name}.tsv", "w") as g:
        for l in open(f"{T}/{name}/rows.jsonl"):
            r = json.loads(l)
            if r.get("path"):
                g.write(r["path"] + "\t" + r["text"] + "\n")
tx = {}
for role in ("s3_train", "s3ov_train"):
    for l in open(f"{R}/{role}.jsonl"):
        if l.strip():
            x = json.loads(l); tx[x.get("take_id", x["video_path"].split("/")[-1][:-4])] = x["transcript"]
for name, hf, cd in (("a_real", f"{T}/heldout.txt", f"{SV}/p1/crop240"), ("c_real", f"{T}/heldout_s3ov.txt", f"{T}/ov/crop")):
    if os.path.exists(hf):
        with open(f"{T}/judge_{name}.tsv", "w") as g:
            for t in open(hf).read().split():
                g.write(f"{cd}/crops/{t}.mp4\t{tx[t]}\n")
PY
cd $C/usr2
for f in $T/judge_*.tsv; do
  n=$(basename $f .tsv)
  $SV/envs/usr2/bin/python $C/judge.py --rows $f --ckpt $SV/models/usr2/vsr_huge_hr.pth --backbone huge --out $T/$n.jsonl
  echo "$n judged: $(wc -l < $T/$n.jsonl)"
done
