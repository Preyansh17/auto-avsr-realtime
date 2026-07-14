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
# One-time setup (MFA is conda-only; separate env, nothing else goes in it).
# MUST use -p on scratch, not -n (name-based envs default to
# ~/.conda/envs, which blows the home quota -- see whisper_rt's venv/conda
# writeup for the same class of trap):
#   module load anaconda3/2025.06
#   export TMPDIR=/scratch/$USER/tmp CONDA_PKGS_DIRS=/scratch/$USER/tmp/conda-pkgs
#   conda create -y -p /scratch/$USER/envs/mfa -c conda-forge montreal-forced-aligner
#   conda run -p /scratch/$USER/envs/mfa mfa model download acoustic english_us_arpa
#   conda run -p /scratch/$USER/envs/mfa mfa model download dictionary english_us_arpa
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
export MFA_ENV="${MFA_ENV:-/scratch/${USER}/envs/mfa}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/tmp}"
mkdir -p "/scratch/${USER}/auto-avsr/output" "${TMPDIR}"

# MFA's default --temporary_directory is ~/Documents/MFA -- SHARED across
# every invocation regardless of corpus. Concurrent `--clean` runs race on
# it (OSError: Stale file handle, killed 2 of 3 concurrent merged-split
# align jobs) and it's on home (quota risk, same class as the pip/conda
# traps elsewhere in this pipeline). Give every job its own scratch dir.
MFA_TEMP="${TMPDIR}/mfa_$(basename "${CORPUS}")_${SLURM_JOB_ID:-$$}"
mkdir -p "${MFA_TEMP}"

echo "Starting MFA align at $(date) on $(hostname): CORPUS=${CORPUS} -> ALIGNED=${ALIGNED}"

# compute nodes don't have `conda` on PATH by default -- module load it here,
# same trap as the CarelessWhisper eval env (login vs compute node PATH
# differs).
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

conda run -p "${MFA_ENV}" mfa align --clean -j "${SLURM_CPUS_PER_TASK}" \
  --temporary_directory "${MFA_TEMP}" \
  "${CORPUS}" english_us_arpa english_us_arpa "${ALIGNED}"

rm -rf "${MFA_TEMP}"

echo "Finished at $(date): $(ls "${ALIGNED}" | grep -c TextGrid || true) TextGrids"
