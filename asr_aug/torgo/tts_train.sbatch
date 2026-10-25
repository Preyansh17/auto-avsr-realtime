#!/bin/bash
# Per-speaker Higgs LoRA for TORGO with the patient B recipe (r64, alpha 128, lr 1e-4, 15-epoch cosine schedule),
# stopped after the pre-registered epoch 8 (the patient's frozen choice; no val-based epoch selection).
#   sbatch --account=<A> -p l40s_public --export=ALL,SPK=F01 tts_train.sbatch
#SBATCH --job-name=torgo_tts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/torgo_aug/logs/%x_%j.out
set -euo pipefail
: "${SPK:?}"
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HIGGS_FT_ROOT=/scratch/pa2753/torgo_aug/higgs/$SPK HIGGS_FT_DATA=/scratch/pa2753/torgo_aug/higgs/$SPK/data
PY=/scratch/pa2753/envs/nemo_asr/bin/python
cd /scratch/pa2753/higgs_ft
echo "SPK=$SPK host=$(hostname) job=$SLURM_JOB_ID $(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY prep_data.py tokenize
EXTRA=""; [[ -f "$HIGGS_FT_ROOT/runs/B_r64/last.pt" ]] && EXTRA="--resume"
$PY train_lora.py --run B_r64 --r 64 --alpha 128 --lr 1e-4 --epochs 15 --stop_after 8 $EXTRA
ls -d "$HIGGS_FT_ROOT/runs/B_r64/epoch_08" && echo TTS_DONE
