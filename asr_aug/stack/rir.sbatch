#!/bin/bash
# Build the room-simulated synthetic clips for arm E2R (rir_synth.py). CPU only.
#   sbatch --account=<A> -p cpu_short rir.sbatch   (any CPU partition)
#SBATCH --job-name=stack_rir
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/pa2753/stack/logs/%x_%j.out
set -euo pipefail
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /scratch/pa2753/stack/code
$PY rir_synth.py build
ln -sfn /scratch/pa2753/stack/rir_audio /scratch/pa2753/stack/root/synth_rir
echo "done $(date)"
