#!/bin/bash
#SBATCH --job-name=carelesswhisper_mfa_align
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_tandon_advanced

# MFA word alignment for CarelessWhisper finetuning (CPU-only, no GPU).
# Input corpus comes from make_carelesswhisper_dataset.py `corpus` mode.
#
# One-time setup (MFA is conda-only; separate env, nothing else goes in it):
#   conda create -n mfa -c conda-forge montreal-forced-aligner
#   conda run -n mfa mfa model download acoustic english_us_arpa
#   conda run -n mfa mfa model download dictionary english_us_arpa
#
#   CORPUS=/scratch/pa2753/carelesswhisper/corpus/legal_train \
#   ALIGNED=/scratch/pa2753/carelesswhisper/aligned/legal_train \
#   sbatch slurm/carelesswhisper_mfa_align.sbatch
#
# Dysarthric speech is exactly what MFA's models were NOT built for -- expect
# some clips to fail alignment. The `csv` step reports what got dropped;
# review that list, don't just proceed.

set -euo pipefail

[[ -n "${CORPUS:-}" ]] || { echo "[FATAL] set CORPUS=<mfa input corpus dir>" >&2; exit 4; }
[[ -n "${ALIGNED:-}" ]] || { echo "[FATAL] set ALIGNED=<textgrid output dir>" >&2; exit 4; }
export MFA_ENV="${MFA_ENV:-mfa}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

echo "Starting MFA align at $(date) on $(hostname): CORPUS=${CORPUS} -> ALIGNED=${ALIGNED}"

conda run -n "${MFA_ENV}" mfa align --clean -j "${SLURM_CPUS_PER_TASK}" \
  "${CORPUS}" english_us_arpa english_us_arpa "${ALIGNED}"

echo "Finished at $(date): $(ls "${ALIGNED}" | grep -c TextGrid || true) TextGrids"
