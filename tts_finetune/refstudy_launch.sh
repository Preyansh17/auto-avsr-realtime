#!/bin/bash
# Experiment RS (reference study, fold 0), scoring only: generation finished earlier. items -> PathBench + neural -> table.
set -euo pipefail
cd /scratch/pa2753/higgs_ft
export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/scratch/pa2753/envs/nemo_asr/bin/python
mkdir -p exp_RS
ids=""
WR="export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /scratch/pa2753/higgs_ft; $PY"
JI=$(sbatch --parsable --account=torch_pr_60_general -p cpu_short --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=00:15:00 --job-name=msRS_items -o logs/msRS_items_%j.out --wrap="$WR ref_study.py items --chunks 3 --allow-incomplete"); echo items=$JI
JP=$(sbatch --parsable --account=torch_pr_60_tandon_advanced -p l40s_public --array=0-2 --dependency=afterany:$JI --export=ALL,FINAL=/scratch/pa2753/higgs_ft/exp_RS final_score.sbatch); echo pb=$JP
JN=$(sbatch --parsable --account=torch_pr_39_general -p l40s_public --array=0-7 --dependency=afterany:$JI --export=ALL,NAMES_FILE=/scratch/pa2753/higgs_ft/exp_RS/neural_todo.txt,NTASKS=8 neural_multi.sbatch); echo neural=$JN
JT=$(sbatch --parsable --account=torch_pr_60_general -p cpu_short --gres=gpu:0 --cpus-per-task=4 --mem=24G --time=00:20:00 --job-name=msRS_table -o logs/msRS_table_%j.out --dependency=afterany:$JP:$JN --wrap="$WR ref_study.py table"); echo table=$JT
echo "$JI $JP $JN $JT" > exp_RS_jobs.txt
