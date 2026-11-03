#!/bin/bash
# G1 = IndexTTS-2 + speaker LoRA for one Higgs voice (asr_aug/gen/README.md): extract features, train the LoRA
# (risingore defaults: r16, lr 5e-5, LoRA+ x8, batch 4, 15 epochs, seed 1; the final epoch is used, fixed in advance like
# Higgs's epoch 8), merge it into the GPT and generate the voice's plan rows. prep_g1.py must have run.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 --export=ALL,VOICE=old|s1|s2 g1.sbatch
# Requeue-safe: each stage is skipped when its output exists. Training has no mid-run resume, but ~800 steps fit well
# inside the 30 min before preemption is allowed, and a rerun is deterministic (seed 1); generation resumes per clip.
#SBATCH --job-name=g1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${VOICE:?}"
G=/scratch/pa2753/gen; D=$G/g1/$VOICE; I=$G/code/index-tts; L=$G/code/index-tts-2-lora; M=$G/models/IndexTTS-2
PY=/scratch/pa2753/envs/idx2/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "VOICE=$VOICE host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
# --val-ratio 0 keeps 1 clip for the tool's val loss (it takes max(1, n * ratio)); Higgs trained on all of tts_train
[ -f $D/feat/medoid_condition.npy ] || $PY $L/extract_codec.py --indextts-dir $I --checkpoint-dir $M --audio-dir $D/wavs \
  --manifest $D/manifest.tsv --output-dir $D/feat --val-ratio 0
[ -f $D/lora/lora_epoch_15.pt ] || $PY $L/train.py --indextts-dir $I --checkpoint-dir $M --data-dir $D/feat \
  --output-dir $D/lora --epochs 15 --batch-size 4 --seed 1
if [ ! -s $D/model/gpt_merged.pth ]; then
  # per-voice model dir: the stock IndexTTS-2 files (symlinked) with the merged GPT and a config pointing at it
  mkdir -p $D/model; for f in $M/*; do b=$(basename $f); [[ $b == gpt.pth || $b == config.yaml ]] || ln -sfn $f $D/model/$b; done
  sed 's/^gpt_checkpoint:.*/gpt_checkpoint: gpt_merged.pth/' $M/config.yaml > $D/model/config.yaml
  $PY $L/merge.py --indextts-dir $I --checkpoint-dir $M --lora-dir $D/lora --epoch 15 --output $D/model/gpt_merged.tmp.pth --no-generate
  mv $D/model/gpt_merged.tmp.pth $D/model/gpt_merged.pth
fi
cd $I
$PY /scratch/pa2753/gen/code/gen_g1.py --voice-dir $D --out-root $G/root/g1
echo "done $(date)"
