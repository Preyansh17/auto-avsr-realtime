#!/bin/bash
#SBATCH --job-name=sv_env_latentsync
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
# Phase 1.5 LatentSync 1.6 env + weights. Drop this job if only KeySync is used (licence decision D1).
# Submit:
#   sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/build_env_latentsync.sbatch
# Pinned: code github.com/bytedance/LatentSync @ a229c3948406bc2cf6eaf4873e662e70c6a04746 (Apache-2.0)
#         weights huggingface.co/ByteDance/LatentSync-1.6 @ c42c7e6c8e9c213626389fa7d9a3c444b8536353 (openrail++)
# Env per setup_env.sh: python 3.10.13 + ffmpeg, requirements.txt (torch 2.5.1 / torchvision 0.20.1 cu121).
# The `sudo apt install libgl1` line is skipped (no root; opencv-python normally finds libGL on the nodes).
set -eo pipefail
SV=/scratch/pa2753/synvid; E=$SV/envs/latentsync; C=$SV/code/LatentSync; M=$SV/models/latentsync
REV=c42c7e6c8e9c213626389fa7d9a3c444b8536353
module purge; module load anaconda3/2026.07
export PIP_CACHE_DIR=$SV/.pipcache CONDA_PKGS_DIRS=$SV/.condapkgs_${SLURM_JOB_NAME:-x} HF_HOME=$SV/.hf
[ -d $C ] || { git clone -q https://github.com/bytedance/LatentSync.git $C && git -C $C checkout -q a229c3948406bc2cf6eaf4873e662e70c6a04746; }
git -C $C log -1 --format="LatentSync %H"
[ -x $E/bin/python ] || conda create -y -p $E -c conda-forge --override-channels python=3.10.13 ffmpeg
P=$E/bin/python
cd $C
$P -m pip install -q -r requirements.txt
# librosa (used by the training data loader) imports pkg_resources, which newer setuptools no longer ships
$P -m pip install -q "setuptools<70"
mkdir -p $M
$E/bin/huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt latentsync_unet.pt --revision $REV --local-dir $M
ln -sfn $M $C/checkpoints
ls -laR $M | grep -v "^total"
$P -c "import torch,diffusers,decord,cv2;print('OK',torch.__version__,diffusers.__version__,cv2.__version__)"
$P -m pip freeze > $SV/envs/latentsync_freeze.txt
