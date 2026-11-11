#!/bin/bash
#SBATCH --job-name=v1_tts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/synvid/v1/logs/%x_%j.out
#SBATCH --open-mode=append
# V1 TTS of the support sentences: Higgs, NW workers on one GPU, then whisper check, then manifest. Resumable.
NW=${NW:-4}; C=/scratch/pa2753/synvid/v1/code; OUT=/scratch/pa2753/synvid/v1/tts
export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/scratch/pa2753/envs/nemo_asr/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
PIDS=""
for i in $(seq 0 $((NW-1))); do $P $C/tts_s.py gen --shard $i --nshard $NW > $OUT/gen_$i.log 2>&1 & PIDS="$PIDS $!"; done
for p in $PIDS; do wait $p || echo "worker $p exit $?"; done
echo "generated: $(ls $OUT/*_16k.wav | wc -l)"
$P $C/tts_s.py check
$P $C/tts_s.py manifest
