#!/bin/bash
#SBATCH --job-name=sv_fetch_text
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
# Phase 2 input: LibriSpeech LM normalised text (openslr resource 11). Submit:
#   sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/fetch_text.sbatch
set -eo pipefail
D=/scratch/pa2753/synvid/data; mkdir -p $D; cd $D
F=librispeech-lm-norm.txt.gz
[ -s $F ] || wget -q -c https://www.openslr.org/resources/11/$F
ls -la $F; sha256sum $F | tee $F.sha256
gzip -t $F && echo "gzip OK"; zcat $F | wc -l
