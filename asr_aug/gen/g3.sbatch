#!/bin/bash
# G3 = VoxCPM2 + LoRA for one Higgs voice (asr_aug/gen/README.md, amendment 4): train the LoRA with the repo's
# finetune script and its LoRA example config (length from prep_g3.py's steps.txt), then generate the voice's plan rows
# with NW sharded workers (HPC cancels low-utilization jobs). prep_g3.py must have run.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 --export=ALL,VOICE=old|s1|s2 g3.sbatch
#   smoke: LIMIT=20 OUT=/scratch/pa2753/gen/root/g3_smoke (generation of 20 rows only)
# Requeue-safe: training auto-resumes from <lora>/latest; it is skipped when step_<N> exists. Generation resumes per clip.
#SBATCH --job-name=g3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${VOICE:?}"
G=/scratch/pa2753/gen; D=$G/g3/$VOICE; V=$G/code/VoxCPM; PY=/scratch/pa2753/envs/voxcpm/bin/python
NW=${NW:-4}; LIMIT=${LIMIT:-0}; OUT=${OUT:-$G/root/g3}
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "VOICE=$VOICE host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
( while sleep 300; do echo "[gpu] $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
MON=$!
read STEPS WARM < $D/steps.txt
if [ ! -d $D/lora/step_$(printf %07d $STEPS) ]; then
  sed -e "s#^pretrained_path:.*#pretrained_path: $G/models/VoxCPM2#" -e "s#^train_manifest:.*#train_manifest: $D/train.jsonl#" \
      -e "s#^num_iters:.*#num_iters: $STEPS#" -e "s#^max_steps:.*#max_steps: $STEPS#" -e "s#^warmup_steps:.*#warmup_steps: $WARM#" \
      -e "s#^valid_interval:.*#valid_interval: 100000#" -e "s#^save_interval:.*#save_interval: 100000#" \
      -e "s#^save_path:.*#save_path: $D/lora#" -e "s#^tensorboard:.*#tensorboard: $D/tb#" \
      $V/conf/voxcpm_v2/voxcpm_finetune_lora.yaml > $D/lora.yaml
  cd $V && $PY scripts/train_voxcpm_finetune.py --config_path $D/lora.yaml
fi
ls $D/lora
pids=()
for k in $(seq 0 $((NW - 1))); do
  $PY $G/code/gen_g3.py --voice $VOICE --lora $D/lora/latest --out-root $OUT --limit $LIMIT --shard $k --nshard $NW \
    >> $G/logs/g3_${SLURM_JOB_ID:-x}_w$k.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null || true
grep -h "generation done\|s/clip\|Error" $G/logs/g3_${SLURM_JOB_ID:-x}_w*.log | tail -8
[ $rc = 0 ]
echo "done $(date)"
