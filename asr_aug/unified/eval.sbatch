#!/bin/bash
# Zero-shot benchmark (PLAN.md 3.3 item 0): unified model offline + buffered streaming at 1.12 s and 0.56 s,
# and nemotron-speech-streaming-en-0.6b offline as the reference, on s1/s2 val/test/oov.
#   sbatch --account=<A> -p l40s_public --export=ALL,PART=offline|stream eval.sbatch
#SBATCH --job-name=unified_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
set -euo pipefail
# NEMO=nemo3 (release 3.0.0, examples from the v3.0.0 tag) or nemo31 (main, examples from the main clone)
NEMO=${NEMO:-nemo3}
PY=/scratch/pa2753/envs/$NEMO/bin/python
REPO=/scratch/pa2753/unified/Speech; [ "$NEMO" = nemo3 ] && REPO=/scratch/pa2753/unified/Speech_v300
T=unified_$NEMO
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# $HOME is at its 30k-file quota: keep every compile cache on scratch
C=/scratch/pa2753/unified/cache; mkdir -p $C
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba
E="/scratch/pa2753/unified/code/unified_eval.py --nemo-repo $REPO"
U=nvidia/nemotron-labs-unified-asr-0.6b
# PART=offline (unified + nemotron_en reference) or PART=stream (unified, 1.12 s and 0.56 s latency)
if [ "${PART:-offline}" = offline ]; then
  $PY $E --model $U --tag $T --mode offline
  $PY $E --model nvidia/nemotron-speech-streaming-en-0.6b --tag nemotron_en_$NEMO --mode offline
else
  $PY $E --model $U --tag $T --mode stream --left 5.6 --chunk 0.56 --right 0.56
  $PY $E --model $U --tag $T --mode stream --left 5.6 --chunk 0.16 --right 0.40
fi
echo "done $(date)"
