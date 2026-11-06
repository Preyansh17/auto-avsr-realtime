#!/bin/bash
#SBATCH --job-name=sv_glsft_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
# A9 G-LSft: LatentSync 1.6 fine-tune with its own training code (scripts/train_unet.py, stage2_512 config) from the
# released latentsync_unet.pt on glsft/affine512 (240 s3_train clips only). lr 1e-5 constant, batch 1, 16 frames,
# STEPS (default 3000) optimisation steps past the released checkpoint's global_step; checkpoint every 500 steps.
# Resumable: a requeued job restarts from the newest checkpoint-*.pt (weights + step; AdamW state is not saved by
# LatentSync, so it restarts fresh). Hard cap 8 GPU-hours over all attempts (glsft/gpu_seconds.txt).
# Needs a GPU with >= 80 GB: sbatch --account=<A> --comment="preemption=yes;requeue=true" \
#   --constraint="a100|h100|h200" glsft_train.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; G=$SV/glsft; L=$SV/code/LatentSync; P=$SV/envs/latentsync/bin/python
STEPS=${STEPS:-3000}; CAP=$((8*3600))
export HF_HOME=$SV/.hf HF_HUB_OFFLINE=1 PATH=$SV/envs/latentsync/bin:$PATH
mkdir -p $G/train $G/cache/embeds $G/cache/mel
USED=$(cat $G/gpu_seconds.txt 2>/dev/null || echo 0)
[ "$USED" -ge "$CAP" ] && { echo "GPU-hour cap reached ($USED s)"; exit 0; }
nvidia-smi --query-gpu=name,memory.total --format=csv
cd $L
S0=$(cat $G/s0.txt 2>/dev/null || $P -c "import torch;print(torch.load('checkpoints/latentsync_unet.pt',mmap=True,map_location='cpu',weights_only=True).get('global_step',0))")
echo $S0 > $G/s0.txt
LAST=$(ls $G/train/*/checkpoints/checkpoint-*.pt 2>/dev/null | awk -F'checkpoint-' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2 || true)
RESUME=${LAST:-checkpoints/latentsync_unet.pt}
echo "released step $S0; resume from $RESUME; target step $((S0+STEPS)); used ${USED}s"
$P - <<EOF
from omegaconf import OmegaConf
c = OmegaConf.load("configs/unet/stage2_512.yaml")
c.data.train_output_dir = "$G/train"
c.data.train_fileslist = ""
c.data.train_data_dir = "$G/affine512"
c.data.audio_embeds_cache_dir = "$G/cache/embeds"
c.data.audio_mel_cache_dir = "$G/cache/mel"
c.data.val_video_path = open("$SV/smoke/template.txt").read().split()[-1]
c.data.val_audio_path = "$SV/smoke/drive_smoke_16k.wav"
c.data.num_workers = 8
c.data.batch_size = 1
c.ckpt.resume_ckpt_path = "$RESUME"
c.ckpt.save_ckpt_steps = 500
c.optimizer.lr = 1e-5
c.run.max_train_steps = $S0 + $STEPS
OmegaConf.save(c, "$G/stage2_512_glsft.yaml")
EOF
grep -n "train_data_dir\|resume_ckpt_path\|max_train_steps\|lr:" $G/stage2_512_glsft.yaml
(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 120; done) &
MON=$!
T0=$(date +%s)
acct() { echo $((USED + $(date +%s) - T0)) > $G/gpu_seconds.txt; }
trap 'acct; kill $MON $TP 2>/dev/null; exit 143' TERM INT
REM=$((CAP-USED))
timeout $REM torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((20000 + SLURM_JOB_ID % 10000)) \
  -m scripts.train_unet --unet_config_path $G/stage2_512_glsft.yaml &
TP=$!
wait $TP || echo "train exit $?"
acct
kill $MON
echo "GPU seconds used in total: $(cat $G/gpu_seconds.txt)"
ls -la $G/train/*/checkpoints/
