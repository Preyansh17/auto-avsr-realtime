#!/bin/bash
# Item 1 biasing grid for one checkpoint: bias_decode.py (beam / prompt / n-gram fusion x clean / words, beam 10),
# then llm_rescore.py (Qwen2.5-1.5B rescoring of the plain-beam 10-best).
#   sbatch --account=<A> -p l40s_public --export=ALL,RUN=/scratch/pa2753/stack/runs/E2_soup bias.sbatch
#SBATCH --job-name=stack_bias
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pa2753/stack/logs/%x_%j.out
set -euo pipefail
: "${RUN:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/pa2753/auto-avsr-realtime
$PY /scratch/pa2753/stack/code/bias_decode.py --model "$RUN/best" --out-dir "$RUN/bias"
$PY /scratch/pa2753/stack/code/llm_rescore.py --bias-dir "$RUN/bias"
echo "done $(date)"
