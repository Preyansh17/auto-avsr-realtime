#!/bin/bash
#SBATCH --job-name=sv_env_usr2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
# Phase 1.1 env for the synvid judge and crop pipeline. Submit:
#   sbatch --account=torch_pr_39_general --partition=cs /scratch/pa2753/synvid/code/build_env_usr2.sbatch
# Builds /scratch/pa2753/synvid/envs/usr2 (python 3.10, torch 2.4.1 cu121, usr2 deps minus mediapipe,
# ibug RetinaFace + FAN) and downloads the USR 2.0 high-resource Large and Huge checkpoints.
set -eo pipefail
SV=/scratch/pa2753/synvid; E=$SV/envs/usr2
module purge; module load anaconda3/2026.07
export PIP_CACHE_DIR=$SV/.pipcache CONDA_PKGS_DIRS=$SV/.condapkgs_${SLURM_JOB_NAME:-x}
[ -x $E/bin/python ] || conda create -y -p $E python=3.10 git-lfs
P=$E/bin/python
$P -m pip install -q torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
$P -m pip install -q "pytorch-lightning>=2.0,<2.5" "hydra-core>=1.3" "omegaconf>=2.3" opencv-python torchmetrics \
   "numpy<2" tqdm av sentencepiece scikit-image scipy gdown
cd $SV/code
[ -d face_detection ] || { GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/hhj1897/face_detection.git && git -C face_detection checkout -q db2a4e8eae8c9c53385ff0773e9db08f03cf21ad; }
W=face_detection/ibug/face_detection/retina_face/weights/Resnet50_Final.pth
[ -s $W ] && [ $(stat -c%s $W) -gt 1000000 ] || \
  wget -q -O $W https://huggingface.co/public-data/ibug-face-detection/resolve/main/retina_face/Resnet50_Final.pth
$P -m pip install -q -e face_detection
[ -d face_alignment ] || { git clone -q https://github.com/hhj1897/face_alignment.git && git -C face_alignment checkout -q 9cf5494e443f26d567972f3f50f6212d65b76c01; }
$P -m pip install -q -e face_alignment
ls -la face_alignment/ibug/face_alignment/fan/weights/ face_detection/ibug/face_detection/retina_face/weights/
(cd face_detection && git log -1 --format="face_detection %H"); (cd face_alignment && git log -1 --format="face_alignment %H")
M=$SV/models/usr2; mkdir -p $M; cd $M
[ -s vsr_large_hr.pth ] || $E/bin/gdown -q 1XDNLfZMv_nn8ALiFmDh3asrJKGC2uHs1 -O vsr_large_hr.pth
[ -s vsr_huge_hr.pth ]  || $E/bin/gdown -q 1LzFOTYu45zCLOHGVLQt7pMGjw6jmmo9Y -O vsr_huge_hr.pth
ls -la $M; sha256sum $M/*.pth
$P -c "import torch,torchvision,ibug.face_detection,ibug.face_alignment,pytorch_lightning as pl,hydra;print('OK',torch.__version__,pl.__version__)"
$P -m pip freeze > $SV/envs/usr2_freeze.txt
