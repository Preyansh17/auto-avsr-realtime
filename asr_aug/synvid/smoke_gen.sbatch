#!/bin/bash
#SBATCH --job-name=sv_smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
# Phase 1.5 smoke test, one generator per job: GEN=latentsync | keysync.
# Template: the longest s3_train clip's 25 fps full-frame copy; driving audio: $SV/smoke/drive_smoke_16k.wav.
# Logs peak GPU memory (nvidia-smi every 2 s) and wall seconds.
# sbatch --account=<A> --constraint="l40s|a100|h100|h200" --export=ALL,GEN=latentsync smoke_gen.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; O=$SV/smoke; mkdir -p $O; export HF_HOME=$SV/.hf
T=$(cat $O/template.txt); A=$O/drive_smoke_16k.wav
echo "GEN=$GEN template=$T audio=$A host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > $O/${GEN}_gpumem.csv &
NV=$!
t0=$(date +%s)
if [ "$GEN" = latentsync ]; then
  cd $SV/code/LatentSync
  export PATH=$SV/envs/latentsync/bin:$PATH
  $SV/envs/latentsync/bin/python -m scripts.inference --unet_config_path configs/unet/stage2_512.yaml \
    --inference_ckpt_path checkpoints/latentsync_unet.pt --inference_steps 20 --guidance_scale 1.5 --enable_deepcache \
    --video_path $T --audio_path $A --video_out_path $O/latentsync_smoke.mp4
  OUT=$O/latentsync_smoke.mp4
else
  cd $SV/code/keysync
  export PATH=$SV/envs/keysync/bin:$PATH
  mkdir -p $O/ks_in/videos $O/ks_in/audios
  cp -f $T $O/ks_in/videos/tpl.mp4; cp -f $A $O/ks_in/audios/tpl.wav
  bash scripts/infer_raw.sh --file_list $O/ks_in/videos/tpl.mp4 --file_list_audio $O/ks_in/audios/tpl.wav \
    --output_folder synvid_smoke --keyframes_ckpt pretrained_models/checkpoints/keyframe_dub.pt \
    --interpolation_ckpt pretrained_models/checkpoints/interpolation_dub.pt --compute_until 45
  OUT=$(ls -t outputs/synvid_smoke/*.mp4 | head -1); cp -f $OUT $O/keysync_smoke.mp4; OUT=$O/keysync_smoke.mp4
fi
t1=$(date +%s); kill $NV
echo "WALL_SECONDS=$((t1-t0)) PEAK_MIB=$(sort -t, -k1 -n $O/${GEN}_gpumem.csv | tail -1 | cut -d, -f1)"
python3 - $OUT <<'PY'
import sys,subprocess,json
print(subprocess.run(["bash","-c",f"/scratch/pa2753/synvid/envs/usr2/bin/python -c \"import cv2;c=cv2.VideoCapture('{sys.argv[1]}');print(c.get(3),c.get(4),c.get(5),c.get(7))\""],capture_output=True,text=True).stdout)
PY
