#!/bin/bash
# Kill-and-resume test for the preemptible recipe (both trainers). Per trainer, three short runs on one GPU:
#   A, A2 = uninterrupted (A vs A2 = the GPU nondeterminism floor);
#   B     = SIGKILLed (whole process group, like a preemption) partway into the epoch after its first resume
#           checkpoint, then rerun with the same command, which resumes.
# Pass: B matches A at least as closely as A2 does (per-epoch val WER, logged losses, final weights).
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" test_resume.sbatch
#SBATCH --job-name=test_resume
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
#SBATCH --open-mode=append
set -uo pipefail
T=/scratch/pa2753/unified/resume_test; ONLY=${ONLY:-}  # ONLY=whisper|nemotron reruns one half, keeping the other's runs
if [ -z "$ONLY" ]; then rm -rf $T; else rm -rf $T/${ONLY:0:1}{A,A2,B}*; fi; mkdir -p $T
echo "host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
cd /home/pa2753/auto-avsr-realtime
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

# run_killed <ready-cmd> <log> <cmd...>: start cmd in its own process group, wait until <ready-cmd> succeeds (a
# complete resume checkpoint exists), let it train 8 s more, SIGKILL the group, then rerun cmd (it must resume).
run_killed() {
  local ready=$1 log=$2; shift 2
  setsid "$@" > $log.part1 2>&1 & local pid=$!
  until eval "$ready"; do sleep 1; kill -0 $pid 2>/dev/null || { echo "FAIL: exited before ckpt"; return 1; }; done
  sleep 8; kill -9 -- -$pid; wait $pid 2>/dev/null; echo "killed pid group $pid at $(date)"
  "$@" > $log.part2 2>&1
}

## Whisper (HF Trainer): whisper-small, 320 rows of E2 seed 1's mix, 3 epochs, the stack recipe otherwise
if [ "$ONLY" != nemotron ]; then
WPY=/scratch/pa2753/envs/whisper_asr/bin/python
head -320 /scratch/pa2753/stack/runs/E2_seed1/train.csv > $T/w_train.csv
wcmd() { echo $WPY -m asr_baselines.whisper_finetune --model openai/whisper-small --root-dir /scratch/pa2753/stack/root \
  --train-file $T/w_train.csv --val-file /scratch/pa2753/multi_patient/labels/s1_val_spm1023.csv --output-dir $T/$1 \
  --label-prefix lang_task --epochs 3 --learning-rate 1e-5 --warmup-steps 5 --weight-decay 0.01 --batch-size 8 \
  --grad-accum 2 --precision bf16 --unfreeze-encoder-layers -1 --num-workers 4 --seed 1 --resume; }
for r in wA wA2; do $(wcmd $r) > $T/$r.log 2>&1; echo "$r exit $?"; done
run_killed 'compgen -G "$T/wB/checkpoint-*/trainer_state.json" > /dev/null' $T/wB.log $(wcmd wB); echo "wB exit $?"
grep -h "\[resume\]" $T/wB.log.part* ; echo
fi

## Nemotron (Lightning): unified model, smoke manifests (20 rows/role), 6 epochs, .ckpt every 2, the ft.sbatch recipe
NPY=/scratch/pa2753/envs/nemo31/bin/python
if [ "$ONLY" != whisper ]; then
export LD_PRELOAD=/scratch/pa2753/envs/nemo31/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13
C=/scratch/pa2753/unified/cache
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba
M=/scratch/pa2753/unified/ft/runs/smoke_unified/manifests
ncmd() { echo $NPY -m asr_baselines.nemotron_finetune --model nvidia/nemotron-labs-unified-asr-0.6b \
  --train-manifest $M/train.json --val-manifest $M/val.json --output-dir $T/$1 --epochs 6 --learning-rate 1e-5 \
  --lr-schedule warmup_flat --warmup-steps 20 --batch-size 4 --grad-accum 1 --unfreeze-encoder-layers -1 \
  --train-decoder --speed-perturb --seed 1 --num-workers 4 --ckpt-every-n-epochs 2 --resume-ckpt auto --freeze-bn-stats; }
for r in nA nA2; do $(ncmd $r) > $T/$r.log 2>&1; echo "$r exit $?"; done
run_killed "$NPY -c 'import glob,sys,zipfile; sys.exit(not any(map(zipfile.is_zipfile, glob.glob(\"$T/nB/lightning_logs/*/checkpoints/*.ckpt\"))))'" $T/nB.log $(ncmd nB); echo "nB exit $?"
grep -h "\[resume\]" $T/nB.log.part* ; echo
fi

$NPY /scratch/pa2753/unified/code/compare_resume.py $T
echo "done $(date)"
