#!/bin/bash
#SBATCH --job-name=v1_sanity
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/pa2753/synvid/v1/logs/%x_%j.out
D=/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105
export USR2_ROOT=$D/code/third_party/usr2_e9e9c94 DASR_SRC=$D/code/hpc/third_party/dasr_metrics PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
/scratch/pa2753/synvid/envs/usr2/bin/python /scratch/pa2753/synvid/v1/code/audio_sanity.py --n 60 2>&1 | grep -v -i hydra
