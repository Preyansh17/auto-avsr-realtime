#!/bin/bash
#SBATCH --job-name=endclip_dec2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/pa2753/synvid/endclip/logs/%x_%j.out
# Step 2 continued: in-home arms b, c (arm a done in job 22501990); s3ov_val arms a, b, c on three readable crop variants.
set -eo pipefail
E=/scratch/pa2753/synvid/endclip; D=/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105
export USR2_ROOT=$D/code/third_party/usr2_e9e9c94 DASR_SRC=$D/code/hpc/third_party/dasr_metrics PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
PY=$E/../envs/usr2/bin/python; CK=$D/model/semg3_usr2_F1_a075_20261027.pth
COMMON="--ckpt $CK --backbone resnet_transformer_huge --modality v --beam 10 --ctc-weight 0.1 --nbest 5 --nbest-keep 20"
nvidia-smi --query-gpu=name --format=csv,noheader; hostname; date
for arm in a:0 b:5 c:10; do n=${arm%%:*}; p=${arm##*:}
  [ $n = a ] || PAD_FRAMES=$p $PY $E/code/usr2_eval_pad.py $COMMON --split-root $E/split --roles dc_home_all --run-dir $E/dec/$n
  for v in fi_v1 fu_v1 fv_v1; do
    PAD_FRAMES=$p $PY $E/code/usr2_eval_pad.py $COMMON --split-root $E/split_ov_$v --roles dc_s3ov_val --run-dir $E/dec_ov/$v/$n
  done; date
done
echo DONE
