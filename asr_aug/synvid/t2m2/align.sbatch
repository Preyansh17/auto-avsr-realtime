#!/bin/bash
#SBATCH --job-name=t2m2_align
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/pa2753/synvid/t2m2/logs/%x_%j.out
# A13 step 0.3a: video CTC forced alignment of the pool (<= 0.5 GPU-h budget). Resumable.
set -eo pipefail
SV=/scratch/pa2753/synvid
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $SV/code/usr2 && $SV/envs/usr2/bin/python $SV/t2m2/code/align_npz.py --pool $SV/t2m2/pool/pool.jsonl \
  --crops $SV/t2m2/crops/npz --ckpt $SV/models/usr2/vsr_huge_hr.pth --out $SV/t2m2/align/video_align.jsonl | tail -5
echo "aligned: $(grep -c words $SV/t2m2/align/video_align.jsonl); errors: $(grep -c error $SV/t2m2/align/video_align.jsonl)"
