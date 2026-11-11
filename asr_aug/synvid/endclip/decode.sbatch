#!/bin/bash
#SBATCH --job-name=endclip_dec
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/pa2753/synvid/endclip/logs/%x_%j.out
# End-of-clip check step 2: delivered v1.8 model alone, beam 10, CTC 0.1, nbest 5 keep 20 (as its eval).
# Arms: a = no pad, b = last frame x5 (0.2 s), c = last frame x10 (0.4 s). Sets: dc_home_all (193), dc_s3ov_val (35).
set -eo pipefail
E=/scratch/pa2753/synvid/endclip; D=/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105
export USR2_ROOT=$D/code/third_party/usr2_e9e9c94 DASR_SRC=$D/code/hpc/third_party/dasr_metrics PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name --format=csv,noheader; hostname; date
for arm in a:0 b:5 c:10; do n=${arm%%:*}; p=${arm##*:}
  [ -f $E/dec/$n/summary.json ] && continue
  PAD_FRAMES=$p $E/../envs/usr2/bin/python $E/code/usr2_eval_pad.py --ckpt $D/model/semg3_usr2_F1_a075_20261027.pth \
    --backbone resnet_transformer_huge --modality v --split-root $E/split --roles dc_home_all dc_s3ov_val \
    --beam 10 --ctc-weight 0.1 --nbest 5 --nbest-keep 20 --run-dir $E/dec/$n
  date
done
echo DONE
