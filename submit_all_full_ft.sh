#!/bin/bash

echo "========================================================"
echo "Submitting 7 Full Fine-tuning Experiments (No LoRA)"
echo "========================================================"
echo ""

# Exp1: All layers (baseline - no freezing)
job1=$(sbatch /home/th3482/auto-avsr/train_patient_av_full_exp1_all.sbatch | awk '{print $4}')
echo "Exp1 - All layers: Job ID $job1"

# Exp2: Fusion only (freeze encoder, aux_encoder, decoder)
job2=$(sbatch /home/th3482/auto-avsr/train_patient_av_full_exp2_fusion.sbatch | awk '{print $4}')
echo "Exp2 - Fusion only: Job ID $job2"

# Exp3: Fusion + Decoder (freeze encoder, aux_encoder)
job3=$(sbatch /home/th3482/auto-avsr/train_patient_av_full_exp3_fusion_decoder.sbatch | awk '{print $4}')
echo "Exp3 - Fusion + Decoder: Job ID $job3"

# Exp4: Last 4 Conformer blocks (freeze first 8 blocks, fusion, decoder)
job4=$(sbatch /home/th3482/auto-avsr/train_patient_av_full_exp4_last4.sbatch | awk '{print $4}')
echo "Exp4 - Last 4 Conformer blocks: Job ID $job4"

# Exp5: Last 4 Conformer blocks + Fusion (freeze first 8 blocks, decoder)
job5=$(sbatch /home/th3482/auto-avsr/train_patient_av_full_exp5_last4_fusion.sbatch | awk '{print $4}')
echo "Exp5 - Last 4 Conformer blocks + Fusion: Job ID $job5"

# Exp6: Last 4 Conformer blocks + Fusion + Decoder (freeze first 8 blocks only)
job6=$(sbatch /home/th3482/auto-avsr/train_patient_av_full_exp6_last4_fusion_dec.sbatch | awk '{print $4}')
echo "Exp6 - Last 4 Conformer blocks + Fusion + Decoder: Job ID $job6"

# Exp7: Decoder only (freeze encoder, aux_encoder, fusion)
job7=$(sbatch /home/th3482/auto-avsr/train_patient_av_full_exp7_decoder.sbatch | awk '{print $4}')
echo "Exp7 - Decoder only: Job ID $job7"

echo ""
echo "========================================================"
echo "All 7 full fine-tuning jobs submitted!"
echo "Job IDs: $job1, $job2, $job3, $job4, $job5, $job6, $job7"
echo "========================================================"
echo ""
echo "Monitor with: squeue -u th3482"
echo "Check W&B: https://wandb.ai"
