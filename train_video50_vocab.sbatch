#!/bin/bash

#SBATCH --job-name=train50
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|v100|a100"
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-user=th3482@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=pr_60_tandon_priority

overlay_ext3=/scratch/th3482/avsr/overlay-15GB-500K.ext3

# Ensure output dir exists
export OUTPUT_DIR=/scratch/th3482/auto-avsr/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

# Experiment directory
EXP_DIR="/scratch/th3482/auto-avsr/experiments"
EXP_NAME="video50_vocab_constrained_bs40_ctc0.1_$(date +%Y%m%d_%H%M%S)"

mkdir -p ${EXP_DIR}

echo "======================================"
echo "Training Configuration:"
echo "  Experiment: ${EXP_NAME}"
echo "  Data: video_50_25p"
echo "  Pretrained: vsr_trlrwlrs2lrs3vox2avsp_base.pth"
echo "  Vocab file: /home/th3482/auto-avsr/sentence.txt"
echo "  Beam size: 40"
echo "  CTC weight: 0.1"
echo "  Pre-beam ratio: 1.5"
echo "======================================"

singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    --writable-tmpfs \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate auto_avsr
cd /home/th3482/auto-avsr
export SSL_CERT_FILE=/home/th3482/auto-avsr/cacert.pem
export HYDRA_FULL_ERROR=1

python train.py \
    hydra.job.chdir=False \
    data.modality='video' \
    data.dataset='video_50' \
    exp_dir='${EXP_DIR}' \
    exp_name='${EXP_NAME}' \
    pretrained_model_path='/home/th3482/auto-avsr/cpts/vsr_trlrwlrs2lrs3vox2avsp_base.pth' \
    vocab_file='/home/th3482/auto-avsr/sentence.txt' \
    beam_size=40 \
    ctc_weight=0.1 \
    pre_beam_ratio=1.5
"

echo "Training completed at $(date)"

