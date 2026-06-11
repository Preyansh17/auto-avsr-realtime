#!/bin/bash

echo "=========================================="
echo "Submitting 6 LoRA Comparison Experiments"
echo "=========================================="
echo ""

# Experiment 1: LoRA on ALL (encoder + aux_encoder + decoder)
echo "Exp 1: LoRA on ALL (encoder + aux_encoder + decoder)"
job1=$(sbatch /home/th3482/auto-avsr/train_patient_av_lora_exp1_all.sbatch | awk '{print $4}')
echo "  Job ID: $job1"
echo ""

# Experiment 2: LoRA on FUSION only
echo "Exp 2: LoRA on FUSION only"
job2=$(sbatch /home/th3482/auto-avsr/train_patient_av_lora_exp2_fusion.sbatch | awk '{print $4}')
echo "  Job ID: $job2"
echo ""

# Experiment 3: LoRA on FUSION + DECODER
echo "Exp 3: LoRA on FUSION + DECODER"
job3=$(sbatch /home/th3482/auto-avsr/train_patient_av_lora_exp3_fusion_decoder.sbatch | awk '{print $4}')
echo "  Job ID: $job3"
echo ""

# Experiment 4: LoRA on last 4 Conformer blocks
echo "Exp 4: LoRA on last 4 Conformer blocks (encoder.encoders.8-11 + aux_encoder.encoders.8-11)"
job4=$(sbatch /home/th3482/auto-avsr/train_patient_av_lora_exp4_last4.sbatch | awk '{print $4}')
echo "  Job ID: $job4"
echo ""

# Experiment 5: LoRA on last 4 Conformer blocks + FUSION
echo "Exp 5: LoRA on last 4 Conformer blocks + FUSION"
job5=$(sbatch /home/th3482/auto-avsr/train_patient_av_lora_exp5_last4_fusion.sbatch | awk '{print $4}')
echo "  Job ID: $job5"
echo ""

# Experiment 6: LoRA on last 4 Conformer blocks + FUSION + DECODER
echo "Exp 6: LoRA on last 4 Conformer blocks + FUSION + DECODER"
job6=$(sbatch /home/th3482/auto-avsr/train_patient_av_lora_exp6_last4_fusion_dec.sbatch | awk '{print $4}')
echo "  Job ID: $job6"
echo ""

echo "=========================================="
echo "All 6 experiments submitted!"
echo "=========================================="
echo ""
echo "Job IDs: $job1, $job2, $job3, $job4, $job5, $job6"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check W&B: https://wandb.ai/th3482-new-york-university/patient_vocab_comparison"
echo ""
echo "Experiment configurations:"
echo "  Exp 1: ALL layers (encoder + aux_encoder + decoder)"
echo "  Exp 2: FUSION only"
echo "  Exp 3: FUSION + DECODER"
echo "  Exp 4: Last 4 Conformer blocks only"
echo "  Exp 5: Last 4 Conformer blocks + FUSION"
echo "  Exp 6: Last 4 Conformer blocks + FUSION + DECODER"

