#!/bin/bash
# G1 / G1d gate (gate_g1.py): judge decodes per voice with the Higgs filter's judges, filter, step-0 diag, gate report.
# whisper_asr env (no B200 kernels): --constraint="l40s|a100|h100|h200". Each stage is skipped when its output exists.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" [--export=ALL,GEN=g1d] gate_g1.sbatch
#SBATCH --job-name=gate_g1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
GEN=${GEN:-g1}; export GEN  # g1, or g1d (README amendment 2)
D=/scratch/pa2753/gen/data/$GEN; C=/scratch/pa2753/gen/code; FL=/scratch/pa2753/gen/fl; [ $GEN = g1 ] || FL=${FL}_$GEN
echo "host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
declare -A J=([old]=/scratch/pa2753/experiments/whisper_asr/whisper_largev3_specaug_splitv1_merged_wdwarmup_seed1/best
              [s1]=/scratch/pa2753/asr_aug/runs/oct1_N_seed1/best [s2]=/scratch/pa2753/multi_patient/runs/whisper_s2_lang_task_seed1/best)
declare -A P=([old]="" [s1]="--prefix lang_task" [s2]="--prefix lang_task")
cd /home/pa2753/auto-avsr-realtime
for v in old s1 s2; do
  [ -s $D/judge_$v.tsv ] || { $PY -m asr_baselines.whisper_eval --model "${J[$v]}" --root-dir $FL/root \
    --label-file $D/synth_all_$v.csv --out $D/judge_$v.tmp.tsv ${P[$v]} && mv $D/judge_$v.tmp.tsv $D/judge_$v.tsv; }
done
$PY $C/gate_g1.py score
[ -s $D/diag_clips.csv ] || $PY $C/gate_g1.py diag --batch 16
$PY $C/gate_g1.py report
echo "done $(date)"
