#!/bin/bash
# LLM rescoring only (llm_rescore.py), for a run whose bias/ already holds the beam 10-best lists.
#   sbatch --account=<A> -p l40s_public --export=ALL,RUN=/scratch/pa2753/stack/runs/E1_seed4 llm.sbatch
#SBATCH --job-name=stack_llm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/stack/logs/%x_%j.out
set -euo pipefail
: "${RUN:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
$PY /scratch/pa2753/stack/code/llm_rescore.py --bias-dir "$RUN/bias"
echo "done $(date)"
