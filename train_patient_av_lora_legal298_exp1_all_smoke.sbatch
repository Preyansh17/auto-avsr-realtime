#!/bin/bash
#SBATCH --job-name=av_lora_legal298_smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-user=th3482@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

set -euo pipefail

echo "Starting Smoke: LoRA on ALL (legal298) at $(date) on $(hostname)"

singularity exec --nv \
  --overlay /scratch/th3482/avsr/overlay-15GB-500K.ext3:ro \
  --writable-tmpfs /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -lc "source /ext3/env.sh; \
  conda activate auto_avsr; \
  export SSL_CERT_FILE=/home/th3482/auto-avsr/cacert.pem; \
  export HYDRA_FULL_ERROR=1; \
  cd /home/th3482/auto-avsr; \
  python train.py \
    hydra.job.chdir=False \
    data.modality='audiovisual' \
    data.dataset='patient_legal298_retinaface_unseen' \
    exp_dir='/scratch/th3482/experiments/patient_legal298_av_lora' \
    exp_name='exp1_lora_all_legal298_smoke_e3' \
    pretrained_model_path='/home/th3482/auto-avsr/cpts/avsr_trlrwlrs2lrs3vox2avsp_base.pth' \
    vocab_file='/scratch/th3482/LipVideoData/patient_legal298_crops_unseen/labels/patient_legal298_sentences.txt' \
    beam_size=40 ctc_weight=0.1 pre_beam_ratio=1.5 \
    trainer.gpus=1 \
    trainer.max_epochs=3 \
    lora.enabled=true lora.r=8 lora.alpha=16 lora.dropout=0.05 \
    'lora.scopes=[encoder,aux_encoder,decoder]' \
    'lora.name_patterns=[]'"

echo "Finished at $(date)"
