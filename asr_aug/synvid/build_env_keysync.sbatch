#!/bin/bash
#SBATCH --job-name=sv_env_keysync
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
# Phase 1.5 KeySync env + weights. Submit:
#   sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/build_env_keysync.sbatch
# Pinned: code github.com/antonibigata/keysync @ 5e2503b171c6ddb4d88cd4cb9702347dcc4aa370 (Apache-2.0)
#         weights huggingface.co/toninio19/keysync @ c26cc4639a109335eac41c398910880d36ef7681 (apache-2.0)
# Env per the KeySync README: python 3.11 + ffmpeg, requirements.txt --no-deps, torch 2.4.1/0.19.1/2.4.1 cu121,
# omegaconf/antlr fix. Run on L40S/A100/H100/H200 later, never B200 (torch 2.4.1 has no sm_100 kernels).
set -eo pipefail
SV=/scratch/pa2753/synvid; E=$SV/envs/keysync; C=$SV/code/keysync; M=$SV/models/keysync
module purge; module load anaconda3/2026.07
export PIP_CACHE_DIR=$SV/.pipcache CONDA_PKGS_DIRS=$SV/.condapkgs_${SLURM_JOB_NAME:-x}
# dlib / face-recognition build from source: use the existing conda toolchain (gcc, g++, cmake)
export PATH=/scratch/pa2753/envs/toolchain/bin:$PATH
[ -d $C ] || { git clone -q https://github.com/antonibigata/keysync.git $C && git -C $C checkout -q 5e2503b171c6ddb4d88cd4cb9702347dcc4aa370; }
git -C $C log -1 --format="keysync %H"
[ -x $E/bin/python ] || conda create -y -p $E -c conda-forge --override-channels python=3.11 ffmpeg git-lfs
P=$E/bin/python
cd $C
# onnxruntime-gpu==1.20.1 (pinned upstream) is no longer on PyPI -> 1.20.2 (same minor); everything else as pinned
sed 's/^onnxruntime-gpu==1.20.1$/onnxruntime-gpu==1.20.2/' requirements.txt > $SV/code/keysync_requirements.txt
$P -m pip install -q -r $SV/code/keysync_requirements.txt --no-deps
$P -m pip install -q torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$P -m pip uninstall -q -y omegaconf antlr4-python3-runtime
$P -m pip install -q "omegaconf==2.3.0" "antlr4-python3-runtime==4.9.3"
# face_alignment is imported by KeySync but missing from its requirements.txt
$P -m pip install -q --no-deps face-alignment==1.4.1
# weights (git-lfs clone of the HF repo, pinned revision), linked where the README expects them
export PATH=$E/bin:$PATH
git lfs install --skip-repo
[ -d $M/.git ] || { GIT_LFS_SKIP_SMUDGE=1 git clone -q https://huggingface.co/toninio19/keysync $M && git -C $M checkout -q c26cc4639a109335eac41c398910880d36ef7681 && git -C $M lfs pull; }
ln -sfn $M $C/pretrained_models
find $M -type f -not -path "*/.git/*" -exec ls -la {} \;
$P -c "import torch,torchvision,torchaudio,diffusers,transformers,xformers;print('OK',torch.__version__,torchvision.__version__,xformers.__version__)"
$P -m pip freeze > $SV/envs/keysync_freeze.txt
