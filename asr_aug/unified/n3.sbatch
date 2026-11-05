#!/bin/bash
# PLAN.md 3.3 item 0, Step 6 (N3): Tianyu's phone recipe, run with his own code (commit 0ded70e) and his exact command
# line, seeds 7 / 1 / 42 together on one GPU (batch 2 alone underuses it and HPC cancels idle GPUs). Requeue-safe: his
# nemotron_finetune resumes from its timed last.ckpt; a seed with finished evaluation is skipped.
#   sbatch --account=<A> --constraint="h100|a100" n3.sbatch
#SBATCH --job-name=n3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
N=/scratch/pa2753/n3
U=$N/code_0ded70e/Dysarthric_ASR/updated
PY=/scratch/pa2753/envs/nemo_asr/bin/python
C=$N/cache; mkdir -p $C $N/runs
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled \
  TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba \
  PYTHONPATH=$U/src
export LD_PRELOAD=/usr/local/cuda/lib64/libnvJitLink.so.13  # nemo_asr env on CUDA 13.3 nodes
[ -s $N/manifests/synth_trainrend_higgsft_raw_s2.jsonl ] || $PY /scratch/pa2753/unified/code/make_n3_extras.py --out $N/manifests
M=$N/manifests
for S in ${SEEDS:-7 1 42}; do
  RD=$N/runs/nem_syn_2b_sc_o189_n189_tts50_seed$S
  [ -s $RD/evaluation/s2ov_test.tsv ] && { echo "seed $S done, skip"; continue; }
  # his lane line, verbatim except --run-dir and the three extra manifests (rebuilt from the package train.csv)
  (cd $U && $PY -m dysarthric_asr.semg.run train --model-type nemotron \
      --split-root /scratch/th3482/Dysarthric_ASR/data/patients/splits/combined_v4 --batch-size 2 --grad-accum 2 \
      --warmup-steps 200 --weight-decay 1e-3 --resume-every-min 5 --nemo-att-context-probs 0.25 0.25 0.25 0.25 \
      --epochs 11 --learning-rate 1e-5 --nemo-lr-schedule cosine \
      --extra-train-manifest $M/synth_wide_higgsft_new150t_s2.jsonl \
      --extra-train-manifest $M/synth_wide_higgszs_med60_nocap_s2.jsonl \
      --extra-train-manifest $M/synth_trainrend_higgsft_raw_s2.jsonl \
      --experiment sc_o189 --subset-seed $S --seed $S --run-dir $RD) > $N/runs/seed$S.log 2>&1 &
done
FAIL=0; for p in $(jobs -p); do wait $p || FAIL=1; done
echo "done $(date) fail=$FAIL"; exit $FAIL
