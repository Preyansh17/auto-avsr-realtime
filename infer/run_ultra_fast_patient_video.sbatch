#!/bin/bash

#SBATCH --job-name=patientVSRM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
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

# ============================================
# Vocabulary Constraint Configuration
# ============================================
# Option 1: No vocabulary constraint (default - allows all words)
# Leave VOCAB_FILE empty or comment out the line below
# VOCAB_FILE=""

# Option 2: Use default vocabulary file (50 legal terms)
# Uncomment the line below to enable vocabulary constraint
VOCAB_FILE="/home/th3482/auto-avsr/infer/50s.txt"

# Option 3: Use custom vocabulary file
# Uncomment and modify the path below
# VOCAB_FILE="/path/to/your/custom/vocab.txt"
# ============================================

singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    --writable-tmpfs \
    /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate auto_avsr
cd /home/th3482/auto-avsr
python infer/batch_inference_ultra_fast_patient_video.py \
  data.modality='video' \
  pretrained_model_path=/home/th3482/auto-avsr/cpts/vsr_trlrwlrs2lrs3vox2avsp_base.pth \
  +detectors='mediapipe' \
  +video_dirs='/scratch/th3482/LipVideoData/patient_25p' \
  +vocab_file='${VOCAB_FILE}' \
  +beam_size=120 \
  +pre_beam_ratio=100.0 \
  +ctc_weight=0.5
"
