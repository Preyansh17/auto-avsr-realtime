#!/bin/bash
# Phase 4 generation: one stripe of final_eval.py's plan_test.json (selected epoch of each run x tts_test x seeds 1-3)
# on one GPU. Same striping as select_gen.sbatch (array task i of NTASKS runs units i, i+NTASKS, ...), so the array can
# be split over accounts:
#   sbatch --account=<A> -p <P> --array=0-2 --export=ALL,PLAN=/scratch/pa2753/higgs_ft/final/plan_test.json,NTASKS=6 final_gen.sbatch
#   sbatch --account=<B> -p <P> --array=3-5 --export=ALL,PLAN=/scratch/pa2753/higgs_ft/final/plan_test.json,NTASKS=6 final_gen.sbatch
# No --account / --partition here (passed on the command line). Resumable. --time suits ~60 clips per task
# (a single run is 120 clips over 6 units); raise it with --time= when NTASKS is small or several runs share a task.
#SBATCH --job-name=ft_final_gen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${PLAN:?set PLAN=<final/plan_test.json>}"
: "${NTASKS:?set NTASKS=<number of stripes>}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "$TASK" -lt "$NTASKS" ] || { echo "array index $TASK >= NTASKS=$NTASKS" >&2; exit 2; }
cd /scratch/pa2753/higgs_ft
echo "host=$(hostname) job=${SLURM_JOB_ID:-} plan=$PLAN task=$TASK/$NTASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/scratch/pa2753/envs/nemo_asr/bin/python gen_ft.py --plan "$PLAN" --task "$TASK" --num-tasks "$NTASKS" ${EXTRA:-}
