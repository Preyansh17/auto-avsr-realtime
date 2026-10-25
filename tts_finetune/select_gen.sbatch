#!/bin/bash
# Phase 3 generation: one stripe of select_epoch.py's plan_val.json on one GPU (gen_ft.py loads every epoch adapter
# side by side and switches per unit). Array task i of NTASKS runs units i, i+NTASKS, ...; NTASKS must be identical in
# every submission of the same plan, so one plan can be spread over several accounts by splitting the --array range:
#   sbatch --account=<A> -p <P> --array=0-1 --export=ALL,PLAN=<plan_val.json>,NTASKS=4 select_gen.sbatch
#   sbatch --account=<B> -p <P> --array=2-3 --export=ALL,PLAN=<plan_val.json>,NTASKS=4 select_gen.sbatch
# No --account / --partition here on purpose (see README: pick the account whose GPU starts soonest). Resumable: clips
# already written are skipped, so a timed-out task can simply be resubmitted. --time is for ~40 clips per task
# (about 5 s each with unmerged LoRA, plus ~3 min model load); raise it with --time= if NTASKS is small.
#SBATCH --job-name=ft_sel_gen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${PLAN:?set PLAN=<runs/RUN/select/plan_val.json>}"
: "${NTASKS:?set NTASKS=<number of stripes>}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "$TASK" -lt "$NTASKS" ] || { echo "array index $TASK >= NTASKS=$NTASKS" >&2; exit 2; }
cd /scratch/pa2753/higgs_ft
echo "host=$(hostname) job=${SLURM_JOB_ID:-} plan=$PLAN task=$TASK/$NTASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/scratch/pa2753/envs/nemo_asr/bin/python gen_ft.py --plan "$PLAN" --task "$TASK" --num-tasks "$NTASKS" ${EXTRA:-}
