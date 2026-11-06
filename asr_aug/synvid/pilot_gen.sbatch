#!/bin/bash
#SBATCH --job-name=sv_pilot
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# Phase 3 pilot for one candidate (GEN=ls | glsft | keysync): generate the 100 pilot clips, crop them with the real
# crop pipeline (crop_pipeline.py), run the judge (USR 2.0 Huge HR). Resumable at every step.
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" \
#   --export=ALL,GEN=ls pilot_gen.sbatch      (glsft also needs CKPT=<checkpoint path>)
set -eo pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; R=$SV/pilot/pilot.jsonl; O=$SV/pilot/$GEN; mkdir -p $O/gen
export HF_HOME=$SV/.hf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
t0=$(date +%s)
if [ "$GEN" = ls ] || [ "$GEN" = glsft ]; then
  CK=${CKPT:-checkpoints/latentsync_unet.pt}
  cd $C/LatentSync; export PATH=$SV/envs/latentsync/bin:$PATH HF_HUB_OFFLINE=1
  $SV/envs/latentsync/bin/python $C/pilot.py ls --rows $R --ckpt $CK --out-dir $O/gen
else
  cd $C/keysync; export PATH=$SV/envs/keysync/bin:$PATH
  K=$O/ks; mkdir -p $K/videos $K/audios $K/out
  : > $K/videos.txt; : > $K/audios.txt
  # KeySync resizes the whole frame to 512x512, so a 16:9 template would be squashed. Give it a square 1080x1080 cut
  # of the 25 fps template, centred on the face (median x of the FAN landmarks from step 1.4), video stream only.
  $SV/envs/usr2/bin/python - <<EOF
import json, os, subprocess
import numpy as np
for l in open("$R"):
    r = json.loads(l); p = r["pid"]
    if os.path.exists(f"$O/gen/{p}.mp4"):
        continue
    sq = f"$K/videos/{p}.mp4"
    if not os.path.exists(sq):
        lm = np.load(f"$SV/p1/crop240/landmarks/{r['template_tid']}.npy")
        x0 = int(min(max(np.nanmedian(lm[:, :, 0]) - 540, 0), 1920 - 1080))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", r["template_path"], "-an", "-vf",
                        f"crop=1080:1080:{x0}:0", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", sq], check=True)
    if not os.path.lexists(f"$K/audios/{p}.wav"):
        os.symlink(r["drive_wav"], f"$K/audios/{p}.wav")
    open("$K/videos.txt", "a").write(f"$K/videos/{p}.mp4\n")
    open("$K/audios.txt", "a").write(f"$K/audios/{p}.wav\n")
EOF
  if [ -s $K/videos.txt ]; then
    bash scripts/infer_raw.sh --file_list $K/videos.txt --file_list_audio $K/audios.txt --output_folder synvid_pilot \
      --keyframes_ckpt pretrained_models/checkpoints/keyframe_dub.pt \
      --interpolation_ckpt pretrained_models/checkpoints/interpolation_dub.pt --compute_until 45
  fi
  for f in outputs/synvid_pilot/*.mp4; do b=$(basename $f .mp4); p=${b%%_*}; [ -f $O/gen/$p.mp4 ] || cp $f $O/gen/$p.mp4; done
fi
t1=$(date +%s)
echo "GEN_WALL_SECONDS=$((t1-t0)) clips=$(ls $O/gen/*.mp4 | wc -l)" | tee -a $O/gen_wall.txt

# crop + judge (rows: take_id = pid, transcript = target sentence)
$SV/envs/usr2/bin/python - <<EOF
import json
with open("$O/crop_rows.jsonl", "w") as f, open("$O/judge_rows.tsv", "w") as g:
    for l in open("$R"):
        r = json.loads(l)
        f.write(json.dumps(dict(take_id=r["pid"], transcript=r["text"], video_path="$O/gen/" + r["pid"] + ".mp4")) + "\n")
        g.write("$O/crop/crops/" + r["pid"] + ".mp4\t" + r["text"] + "\n")
EOF
PREP_ROOT=$C/repo_prep $SV/envs/usr2/bin/python $C/crop_pipeline.py --rows $O/crop_rows.jsonl --out-dir $O/crop \
  --src-dir $O/gen
$SV/envs/usr2/bin/python - <<EOF
import json, os
with open("$O/judge_rows_ok.tsv", "w") as g:
    for l in open("$O/judge_rows.tsv"):
        if os.path.exists(l.split("\t")[0]):
            g.write(l)
EOF
cd $C/usr2
$SV/envs/usr2/bin/python $C/judge.py --rows $O/judge_rows_ok.tsv --ckpt $SV/models/usr2/vsr_huge_hr.pth \
  --backbone huge --out $O/judge.jsonl
echo "judged: $(wc -l < $O/judge.jsonl)"
