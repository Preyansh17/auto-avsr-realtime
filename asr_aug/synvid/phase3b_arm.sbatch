#!/bin/bash
#SBATCH --job-name=sv_3b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# Phase 3b, one arm (ARM=rate_a | rate_b | own; rows from phase3b_audio.py): G-LSft generation, then the same crop
# pipeline and judge (USR 2.0 Huge HR) as pilot_gen.sbatch. Resumable at every step.
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" \
#   --export=ALL,ARM=rate_a phase3b_arm.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; O=$SV/phase3b/$ARM; R=$O/rows.jsonl; mkdir -p $O/gen
CK=${CKPT:-$SV/glsft/train/train-2026_11_06-01:25:40/checkpoints/checkpoint-3000.pt}
export HF_HOME=$SV/.hf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
(cd $C/LatentSync && PATH=$SV/envs/latentsync/bin:$PATH HF_HUB_OFFLINE=1 \
  $SV/envs/latentsync/bin/python $C/pilot.py ls --rows $R --ckpt $CK --out-dir $O/gen)
echo "generated: $(ls $O/gen/*.mp4 | wc -l) / $(wc -l < $R)"

$SV/envs/usr2/bin/python - <<EOF
import json
with open("$O/crop_rows.jsonl", "w") as f, open("$O/judge_rows.tsv", "w") as g:
    for l in open("$R"):
        r = json.loads(l)
        f.write(json.dumps(dict(take_id=r["pid"], transcript=r["text"], video_path="$O/gen/" + r["pid"] + ".mp4")) + "\n")
        g.write("$O/crop/crops/" + r["pid"] + ".mp4\t" + r["text"] + "\n")
EOF
PREP_ROOT=$C/repo_prep $SV/envs/usr2/bin/python $C/crop_pipeline.py --rows $O/crop_rows.jsonl --out-dir $O/crop \
  --src-dir $O/gen
$SV/envs/usr2/bin/python - <<EOF
import os
with open("$O/judge_rows_ok.tsv", "w") as g:
    for l in open("$O/judge_rows.tsv"):
        if os.path.exists(l.split("\t")[0]):
            g.write(l)
EOF
cd $C/usr2
$SV/envs/usr2/bin/python $C/judge.py --rows $O/judge_rows_ok.tsv --ckpt $SV/models/usr2/vsr_huge_hr.pth \
  --backbone huge --out $O/judge.jsonl
echo "judged: $(wc -l < $O/judge.jsonl)"
