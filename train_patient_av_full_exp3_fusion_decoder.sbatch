#!/bin/bash
#SBATCH --job-name=av_full_exp3_fusion_dec
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-user=th3482@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=pr_60_tandon_priority

set -euo pipefail

echo "Starting Full FT Exp3: Fusion + Decoder at $(date) on $(hostname)"

singularity exec --nv \
  --overlay /scratch/th3482/avsr/overlay-15GB-500K.ext3:ro \
  --writable-tmpfs /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -lc "source /ext3/env.sh; \
  conda activate auto_avsr; \
  export SSL_CERT_FILE=/home/th3482/auto-avsr/cacert.pem; \
  export HYDRA_FULL_ERROR=1; \
  cd /home/th3482/auto-avsr; \
  python train.py \
    hydra.job.chdir=False \
    data.modality='audiovisual' \
    data.dataset='patient_retinaface_unseen' \
    data.train_limit_n=90 \
    exp_dir='/scratch/th3482/experiments/patient_av_full_comparison' \
    exp_name='full_exp3_fusion_decoder' \
    pretrained_model_path='/home/th3482/auto-avsr/cpts/avsr_trlrwlrs2lrs3vox2avsp_base.pth' \
    vocab_file='/home/th3482/auto-avsr/infer/50s.txt' \
    beam_size=40 ctc_weight=0.1 pre_beam_ratio=1.5 \
    trainer.gpus=1 \
    lora.enabled=false \
    freeze.enabled=true \
    'freeze.scopes=[encoder,aux_encoder]'"

echo "Finished at $(date)"
