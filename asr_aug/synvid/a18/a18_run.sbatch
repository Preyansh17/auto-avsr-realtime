#!/bin/bash
#SBATCH --job-name=a18
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pa2753/synvid/a18/logs/%x_%j.out
#SBATCH --open-mode=append
# A18 generated-video arm $ARM (manifest $MAN) for one seed. Resumable.
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="a100|h100|h200" --export=ALL,SEED=1 a18_run.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; V=$SV/v0; C=$SV/v1/code; PY=$SV/envs/usr2/bin/python
D=/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105
export USR2_ROOT=$D/code/third_party/usr2_e9e9c94 DASR_SRC=$D/code/hpc/third_party/dasr_metrics PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
CK=$SV/models/usr2/vsr_huge_hr.pth; SR=$V/split; SEED=${SEED:?}; ARM=${ARM:?}; MAN=${MAN:?}; R=$V/runs/$ARM/seed$SEED; mkdir -p $R
nvidia-smi --query-gpu=name --format=csv,noheader
cd $USR2_ROOT
if [ ! -f $R/train_done.json ]; then
  $PY $C/usr2_ft_v1.py train --ckpt $CK --split-root $SR --run-dir $R --seg longest --updates 1000 --eval-every 100 \
    --warmup 100 --accum 4 --lr 2e-5 --seed $SEED --video-aug --aug-affine 4,3,0.05 --aug-photo --aug-speed 0.85,1.15 \
    --aug-trunc 0.3 --train-role v0_A0_train --extra-train-jsonl $MAN --val-role v0_val --resume-every-min 5
fi
[ -f $R/wise_a0.75.pth ] || $PY $C/usr2_ft_v1.py wise --ckpt $CK --split-root $SR --run-dir $R --ft $R/best.pth --alpha 0.75
for a in 1.0 0.75; do
  M=$R/best.pth; [ $a = 0.75 ] && M=$R/wise_a0.75.pth
  [ -f $R/a$a/summary.json ] || $PY $C/usr2_eval.py --ckpt $M --backbone resnet_transformer_huge --modality v \
    --split-root $SR --beam 10 --ctc-weight 0.1 --nbest 5 --nbest-keep 20 --roles v0_T v0_val --run-dir $R/a$a
done
rm -f $R/wise_a0.75.pth $R/best.pth $R/resume*.pt
echo "arm $ARM seed $SEED done"
