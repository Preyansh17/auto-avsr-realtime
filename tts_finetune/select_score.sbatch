#!/bin/bash
# Phase 3 scoring + selection for one run, one GPU job:
#   select_epoch.py score  = prep (completeness check, PathBench items) -> voice (TitaNet/WavLM via
#                            tts_eval/neural_metrics_v2.Scorer) -> PathBench (pb_eval/pb_score_items.py, no --dartp) -> aggregate
#   select_epoch.py select = pre-registered rule -> runs/$RUN/select.md and select.json
# Submit after every select_gen task finished; account/partition on the command line:
#   sbatch --account=<A> -p <P> --dependency=afterok:<gen job ids> --export=ALL,RUN=A_r16 select_score.sbatch
# Env follows tts_eval/v2/run_env.sh (nemo_asr python, offline HF, NO higgs_site: nemo/HuBERT need the stock transformers);
# the PathBench step sources pb_eval/pb_env.sh itself. ~180 clips: a few minutes of work plus model loads.
#SBATCH --job-name=ft_sel_score
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%j.out
set -euo pipefail
: "${RUN:?set RUN=<run name, e.g. A_r16>}"
PY=/scratch/pa2753/envs/nemo_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /scratch/pa2753/higgs_ft
echo "host=$(hostname) job=${SLURM_JOB_ID:-} run=$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" select_epoch.py score --run "$RUN" ${EXTRA:-}
rc=0
"$PY" select_epoch.py select --run "$RUN" || rc=$?
if [ "$rc" -eq 3 ]; then
  echo "NO ELIGIBLE EPOCH for $RUN: every epoch violates the ramble or voice constraint (see runs/$RUN/select.md)"
elif [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
