#!/bin/bash
# G1d = G1 with duration control (asr_aug/gen/README.md, amendment 2): the merged G1 model for one voice regenerates its
# plan rows with the s2mel length scaled by a fixed per-voice factor. g1.sbatch must have built $D/model.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 --export=ALL,VOICE=old|s1|s2 g1d.sbatch
#   LIMIT=20 OUT=/scratch/pa2753/gen/root/g1d_smoke for the smoke. Generation resumes per clip.
#SBATCH --job-name=g1d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${VOICE:?}"
G=/scratch/pa2753/gen; D=$G/g1/$VOICE; I=$G/code/index-tts; OUT=${OUT:-$G/root/g1d}; LIMIT=${LIMIT:-0}
declare -A SCALE=([old]=0.586 [s1]=0.528 [s2]=0.483)  # real-train median sec/word / G1 median sec/word, fixed in advance
PY=/scratch/pa2753/envs/idx2/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "VOICE=$VOICE scale=${SCALE[$VOICE]} out=$OUT host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
# one-line patch: the s2mel length takes an optional per-instance factor (default 1.0 = stock behaviour)
grep -q 'getattr(self, "dur_scale", 1.0)' $I/indextts/infer_v2.py || \
  sed -i 's/target_lengths = (code_lens \* 1\.72)\.long()/target_lengths = (code_lens * 1.72 * getattr(self, "dur_scale", 1.0)).long()/' $I/indextts/infer_v2.py
grep -q 'getattr(self, "dur_scale", 1.0)' $I/indextts/infer_v2.py
cd $I
# NW workers share the GPU: one batch-1 TTS process left the B200 mostly idle and HPC cancelled job 21613869 for low
# GPU utilization (Nov 1). Each worker takes rows shard::NW; nvidia-smi is sampled every 5 min into the job log.
NW=${NW:-4}; C=${SLURM_CPUS_PER_TASK:-8}; export OMP_NUM_THREADS=$(( C / NW > 0 ? C / NW : 1 ))
( while sleep 300; do echo "[gpu] $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
MON=$!
pids=()
for k in $(seq 0 $((NW - 1))); do
  $PY $G/code/gen_g1.py --voice-dir $D --out-root $OUT --dur-scale ${SCALE[$VOICE]} --limit $LIMIT --shard $k --nshard $NW &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null || true
[ $rc = 0 ]
echo "done $(date)"
