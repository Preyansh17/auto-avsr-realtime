#!/bin/bash
# Clean-split Higgs adapter for the re-voicing study (2026-10-20): the patient B recipe (r64, alpha 128, lr 1e-4,
# 15-epoch cosine, epoch 8 fixed in advance) on a TTS split that drops every ASR val AND test sentence
# (tts_finetune/prep_data.py default --asr-holdout val,test). The frozen B_r64 adapter dropped ASR test only.
#   sbatch --account=<A> tts_clean.sbatch
#SBATCH --job-name=rv_tts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/asr_aug/revoice/logs/%x_%j.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HIGGS_FT_ROOT=/scratch/pa2753/higgs_ft_clean HIGGS_FT_DATA=/scratch/pa2753/higgs_ft_clean/data \
  HIGGS_FT_REFS=/scratch/pa2753/higgs_ft_clean/refs GIT_HASH=${GIT_HASH:-unknown}
PY=/scratch/pa2753/envs/nemo_asr/bin/python
cd /scratch/pa2753/higgs_ft
echo "host=$(hostname) job=${SLURM_JOB_ID:-} $(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ -s $HIGGS_FT_DATA/splits.json ]] || $PY prep_data.py split
$PY - <<'EOF'
import json
l = json.load(open("/scratch/pa2753/higgs_ft_clean/data/splits.json"))["log"]
assert sorted(l["asr_holdout"]) == ["test", "val"], l["asr_holdout"]
print("clean split", l["split_hash"], l["clips"], "dropped", l["asr_holdout_clips_dropped"])
EOF
$PY prep_data.py tokenize
EXTRA=""; [[ -f "$HIGGS_FT_ROOT/runs/B_r64/last.pt" ]] && EXTRA="--resume"
$PY train_lora.py --run B_r64 --r 64 --alpha 128 --lr 1e-4 --epochs 15 --stop_after 8 $EXTRA
ls -d "$HIGGS_FT_ROOT/runs/B_r64/epoch_08" && echo TTS_DONE
