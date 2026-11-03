#!/bin/bash
# Step 0 of PLAN.md 3.3 item 5: zero-shot large-v3 + WavLM-SV over real vs Higgs synthetic clips (diag_synth.py).
#   sbatch --account=<A> -p a100_tandon,l40s_public diag.sbatch
#SBATCH --job-name=stack_diag
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/pa2753/stack/logs/%x_%j.out
set -euo pipefail
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
$PY /scratch/pa2753/stack/code/diag_synth.py run --batch 16
$PY /scratch/pa2753/stack/code/diag_synth.py report
