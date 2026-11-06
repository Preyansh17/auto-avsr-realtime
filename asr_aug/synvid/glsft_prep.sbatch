#!/bin/bash
#SBATCH --job-name=sv_glsft_prep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# A9 G-LSft training pairs from the 240 s3_train clips (never any other role). Needs drive/train (make_drive.py
# gen --set train + check) to exist. Every step is resumable.
#  1 word timings on the real clip: align_words.py video (judge = USR 2.0 Huge HR, video CTC head, Tianyu's crop)
#  2 word timings on the healthy TTS clip of the same transcript: align_words.py audio (hubert-large-ls960-ft)
#  3 piecewise retime of the TTS clip onto the clip's word timings (align_words.py retime)
#  4 mux the retimed wav into the 25 fps full-frame clip (A1 template file; video stream copied, not re-encoded)
#  5 LatentSync's own 512 px affine face alignment (preprocess/affine_transform.py) -> glsft/affine512/*.mp4
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" glsft_prep.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; G=$SV/glsft; D=$SV/drive/train
ROLES=/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles
mkdir -p $G/retimed $G/muxed $G/affine512
export PATH=$PATH:$SV/envs/latentsync/bin   # ffmpeg only (appended; pythons are called by absolute path)

# 1 video word timings
(cd $C/usr2 && $SV/envs/usr2/bin/python $C/align_words.py video --rows $ROLES/s3_train.jsonl \
   --ckpt $SV/models/usr2/vsr_huge_hr.pth --backbone huge --out $G/video_align.jsonl) > $G/video_align.log 2>&1
echo "video aligned: $(grep -c words $G/video_align.jsonl) / errors: $(grep -c error $G/video_align.jsonl || true)"

# 2 audio word timings (only TTS clips that passed the whisper check, so the words are right)
/scratch/pa2753/envs/nemo_asr/bin/python - <<EOF
import json
with open("$G/audio_items.tsv", "w") as f:
    for l in open("$D/whisper.jsonl"):
        r = json.loads(l)
        if r["pass"]:
            f.write(f"{r['id']}\t{r['wav']}\t{r['text']}\n")
EOF
PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 \
  /scratch/pa2753/envs/nemo_asr/bin/python $C/align_words.py audio --items $G/audio_items.tsv \
  --out $G/audio_align.jsonl > $G/audio_align.log 2>&1
echo "audio aligned: $(grep -c words $G/audio_align.jsonl)"

# 3 retime (TTS id == take_id for the train set)
awk -F'\t' '{print $1, $1}' $G/audio_items.tsv > $G/pairs.txt
/scratch/pa2753/envs/nemo_asr/bin/python $C/align_words.py retime --video-align $G/video_align.jsonl \
  --audio-align $G/audio_align.jsonl --pairs $G/pairs.txt --out-dir $G/retimed --out $G/retime.jsonl > $G/retime.log 2>&1
echo "retimed: $(ls $G/retimed | wc -l); skipped: $(grep -c skip $G/retime.jsonl || true)"

# 4 mux into the 25 fps full-frame clip
/scratch/pa2753/envs/nemo_asr/bin/python - <<EOF
import json, os, subprocess
rows = {}
for l in open("$ROLES/s3_train.jsonl"):
    if l.strip():
        x = json.loads(l); rows[x.get("take_id", os.path.basename(x["video_path"])[:-4])] = x
n = 0
for l in open("$G/retime.jsonl"):
    r = json.loads(l)
    if "wav" not in r:
        continue
    tid = r["take_id"]; out = f"$G/muxed/{tid}.mp4"
    if os.path.exists(out):
        n += 1; continue
    real = os.path.realpath(rows[tid]["video_path"])
    src = real.split("/processed/v1/")[0] + "/processed/v1/_normalized_25p/" + tid + ".mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-i", r["wav"], "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out + ".tmp.mp4"], check=True)
    os.replace(out + ".tmp.mp4", out); n += 1
print("muxed", n)
EOF

# 5 LatentSync affine alignment at 512
cd $C/LatentSync
$SV/envs/latentsync/bin/python - <<EOF
import preprocess.affine_transform as A
A.affine_transform_multi_gpus("$G/muxed", "$G/affine512", "$G/affine_tmp", 512, 2)
EOF
echo "affine512: $(ls $G/affine512/*.mp4 2>/dev/null | wc -l)"
