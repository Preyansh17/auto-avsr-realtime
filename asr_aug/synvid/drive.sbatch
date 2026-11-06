#!/bin/bash
#SBATCH --job-name=sv_drive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# Phase 2.2 driving audio: Higgs, NW parallel workers on one GPU (keeps utilisation up), then whisper check.
# Resumable. sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" drive.sbatch
set -o pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; NW=${NW:-4}
export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/scratch/pa2753/envs/nemo_asr/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 60; done) &
MON=$!
for SET in train pool; do
  OUT=$SV/drive/$SET; mkdir -p $OUT
  PIDS=""
  for i in $(seq 0 $((NW-1))); do
    $P $C/make_drive.py gen --set $SET --out $OUT --shard $i --nshard $NW > $OUT/gen_$i.log 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in $PIDS; do wait $p || echo "worker $p exit $?"; done
  echo "$SET generated: $(ls $OUT/*_16k.wav | wc -l)"
  $P $C/make_drive.py check --dir $OUT --out $OUT/whisper.jsonl
  $P - <<EOF
import json; r=[json.loads(l) for l in open("$OUT/whisper.jsonl")]
print("$SET whisper: n=%d pass=%d" % (len(r), sum(x["pass"] for x in r)))
EOF
done
# word timings of the pool clips that passed (for 2.3 rate match; drive_final.py runs after step 1.4)
OUT=$SV/drive/pool
$P -c "
import json
with open('$OUT/align_items.tsv', 'w') as f:
    for l in open('$OUT/whisper.jsonl'):
        r = json.loads(l)
        if r['pass']:
            f.write(r['id'] + '\t' + r['wav'] + '\t' + r['text'] + '\n')
"
$P $C/align_words.py audio --items $OUT/align_items.tsv --out $OUT/audio_align.jsonl > $OUT/audio_align.log 2>&1
echo "pool aligned: $(grep -c words $OUT/audio_align.jsonl)"
kill $MON
