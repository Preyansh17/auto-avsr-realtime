#!/bin/bash
# Fine-tune a NeMo RNNT base on the E2 real data (old + s1 + s2 train, 794 clips), then decode best.nemo zero-shot-style
# with unified_eval.py on s1/s2 val/test/oov + s2ov_test (PLAN.md 3.3 item 0, step 2). Same recipe for both bases.
#   sbatch --account=<A> -p a100_tandon,l40s_public --export=ALL,BASE=unified|nemotron_en,SEED=1 ft.sbatch
#   SEEDS="1 2 3" instead of SEED: train the seeds side by side on the one GPU (~14 GB each). One seed alone used ~42% of
#   the GPU, under HPC's 50% cancellation line, and packing also keeps us under the per-user GPU cap. Recipe unchanged.
#   SMOKE=1: 20 rows per role, 1 epoch.
#   RECIPE=e2 (default: old + s1 + s2 train, 794 clips), v2r (semg2 v2, real only: s2ov_train + s2_train, 429 clips,
#   val s2ov_val + s2_val) v2h (v2r + Tianyu's 5,378 Higgs rows) or v2o (v2r + our 1,749 semg2 Higgs clips); run dirs <base>_<recipe>_seed<N>, scored with teval.sh.
#   STREAM_P=0.5 (unified only): streaming-aware training (nemotron_finetune --unified-stream-prob); run dir gets _st.
# Preemptible: sbatch --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" ... (no -p). Each
# seed writes a resume .ckpt every CKPT_EVERY epochs (default 2; ~7 GB, one rolling file) and a requeued job resumes
# it (--resume-ckpt auto: model, optimizer, scheduler, loop, best-WER and RNG states); a seed with train_done skips
# training. Logs append across requeues.
#SBATCH --job-name=unified_ft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${BASE:?}"; SEEDS=${SEEDS:-${SEED:?}}; NS=$(wc -w <<<"$SEEDS")
case $BASE in
  unified) MODEL=nvidia/nemotron-labs-unified-asr-0.6b ;;
  nemotron_en) MODEL=nvidia/nemotron-speech-streaming-en-0.6b ;;
  *) echo "bad BASE"; exit 2 ;;
esac
SMOKE=${SMOKE:-0}; RECIPE=${RECIPE:-e2}
EPOCHS=${EPOCHS:-60}; [ "$SMOKE" = 1 ] && EPOCHS=1
PY=/scratch/pa2753/envs/nemo31/bin/python
C=/scratch/pa2753/unified/cache; mkdir -p $C
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba
# numba-cuda JITs with the env's NVVM 13.4; the system nvJitLink (13.3) is too old for it (ERROR_OUTDATED_LIBRARY),
# so preload the env's matching nvJitLink instead of the system one used for the nemo_asr env.
JL=/scratch/pa2753/envs/nemo31/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13
[[ -n "$JL" ]] && export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
run_seed() {
  SEED=$1
  TAG=${BASE}_ft_seed${SEED}; [ "$RECIPE" = e2 ] || TAG=${BASE}_${RECIPE}${STREAM_P:+_st}_seed${SEED}; [ "$SMOKE" = 1 ] && TAG=smoke_${BASE}${SMOKE_SUFFIX:-}  # SMOKE_SUFFIX: keep a new smoke apart from older smoke dirs
  OUT=/scratch/pa2753/unified/ft/runs/$TAG; M=$OUT/manifests; mkdir -p $OUT
  echo "BASE=$BASE MODEL=$MODEL SEED=$SEED SEEDS=[$SEEDS] EPOCHS=$EPOCHS host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
  $PY /scratch/pa2753/unified/code/make_ft_manifests.py --out $M --recipe $RECIPE $([ "$SMOKE" = 1 ] && echo --limit 20)
  cd /home/pa2753/auto-avsr-realtime
  # Recipe (both bases): full fine-tune, batch 4 x accum 1 (effective 4; was 2 x 2, raised for GPU utilisation), warmup_flat lr 1e-5 (the corrected schedule that tied the
  # legacy noam run, asr_aug/revoice/README.md amendment 3), speed perturb, SpecAugment defaults, val-WER selection.
  [ -f $OUT/train_done ] || $PY -m asr_baselines.nemotron_finetune --model $MODEL --train-manifest $M/train.json --val-manifest $M/val.json \
    --output-dir $OUT --epochs $EPOCHS --learning-rate 1e-5 --lr-schedule warmup_flat --warmup-steps 200 \
    --batch-size 4 --grad-accum 1 --unfreeze-encoder-layers -1 --train-decoder --speed-perturb --seed $SEED \
    --num-workers $((16 / NS)) --ckpt-every-n-epochs ${CKPT_EVERY:-2} --resume-ckpt auto \
    $([ "${FREEZE_BN:-1}" = 1 ] && echo --freeze-bn-stats) ${STREAM_P:+--unified-stream-prob $STREAM_P}
  touch $OUT/train_done
  B=$OUT/nemotron_best.nemo; [ -s $B ] || B=$OUT/nemotron_patient.nemo
  if [ "$RECIPE" != e2 ]; then bash /scratch/pa2753/unified/code/teval.sh $B $TAG $BASE $OUT/teval
  else
  $PY /scratch/pa2753/unified/code/unified_eval.py --model $B --tag $TAG --mode offline --out $OUT/eval
  # streaming needs the NeMo 3.1 checkout (= installed nemo31); unified_eval's default is the 3.0.0 one, whose
  # streaming script has no target_lang ("Key 'target_lang' not in 'TranscriptionConfig'")
  [ "$BASE" = unified ] && [ "$SMOKE" = 0 ] && $PY /scratch/pa2753/unified/code/unified_eval.py --model $B --tag $TAG \
    --nemo-repo /scratch/pa2753/unified/Speech \
    --mode stream --left 5.6 --chunk 0.56 --right 0.56 --out $OUT/eval
  fi
  rm -rf $OUT/lightning_logs/*/checkpoints
}
for S in $SEEDS; do
  if [ "$NS" = 1 ]; then run_seed $S; else run_seed $S >> /scratch/pa2753/unified/logs/${SLURM_JOB_NAME:-ft}_${SLURM_JOB_ID:-0}_seed$S.out 2>&1 &  # >> : requeues append
    sleep 90; fi  # stagger so the seeds don't race on the shared numba/triton caches
done
wait
for S in $SEEDS; do [ "$NS" = 1 ] || grep -a -E "^BASE|Saved finetuned model \(best|WER [0-9]" /scratch/pa2753/unified/logs/${SLURM_JOB_NAME:-ft}_${SLURM_JOB_ID:-0}_seed$S.out || true; done
echo "done $(date)"
