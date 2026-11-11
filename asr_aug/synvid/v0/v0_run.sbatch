#!/bin/bash
#SBATCH --job-name=v0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96GB
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/pa2753/synvid/v0/logs/%x_%j.out
#SBATCH --open-mode=append
# A14 V0 real-video oracle: arms A0 and A1 for one seed, side by side on one GPU. Resumable (requeue safe).
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="a100|h100|h200" --export=ALL,SEED=1 v0_run.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; V=$SV/v0; PY=$SV/envs/usr2/bin/python
D=/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105
export USR2_ROOT=$D/code/third_party/usr2_e9e9c94 DASR_SRC=$D/code/hpc/third_party/dasr_metrics PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
CK=$SV/models/usr2/vsr_huge_hr.pth; SR=$V/split; SEED=${SEED:?}; UPD=${UPD:-1000}; EV=${EV:-100}; WU=${WU:-100}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
arm() {
  local R=$V/runs/$1/seed$SEED; mkdir -p $R
  if [ ! -f $R/train_done.json ]; then
    $PY $V/code/usr2_ft.py train --ckpt $CK --split-root $SR --run-dir $R --seg longest --updates $UPD --eval-every $EV \
      --warmup $WU --accum 4 --lr 2e-5 --seed $SEED --video-aug --aug-affine 4,3,0.05 --aug-photo --aug-speed 0.85,1.15 \
      --aug-trunc 0.3 --train-role v0_${1}_train --val-role v0_val --resume-every-min 5
  fi
  [ -f $R/wise_a0.75.pth ] || $PY $V/code/usr2_ft.py wise --ckpt $CK --split-root $SR --run-dir $R --ft $R/best.pth --alpha 0.75
  for a in 1.0 0.75; do
    M=$R/best.pth; [ $a = 0.75 ] && M=$R/wise_a0.75.pth
    [ -f $R/a$a/summary.json ] || $PY $V/code/usr2_eval.py --ckpt $M --backbone resnet_transformer_huge --modality v \
      --split-root $SR --beam 10 --ctc-weight 0.1 --nbest 5 --nbest-keep 20 --roles v0_T v0_val --run-dir $R/a$a
  done
  rm -f $R/wise_a0.75.pth
  echo "arm $1 seed $SEED done"
}
cd $USR2_ROOT
arm A0 > $V/logs/A0_seed${SEED}.log 2>&1 &
arm A1 > $V/logs/A1_seed${SEED}.log 2>&1 &
st=0; for j in $(jobs -p); do wait $j || st=1; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
tail -n 3 $V/logs/A0_seed${SEED}.log $V/logs/A1_seed${SEED}.log; exit $st
