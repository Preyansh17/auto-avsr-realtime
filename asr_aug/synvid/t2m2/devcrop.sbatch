#!/bin/bash
#SBATCH --job-name=t2m2_devcrop
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/synvid/t2m2/logs/%x_%j.out
# A13 step 0.2: deliverable devcrop.py (phone front end, MediaPipe 0.10.21, same args as synvid v18crop.sbatch) on every
# pool clip's source video. 16 CPU shards, resumable (existing npz skipped).
set -o pipefail
SV=/scratch/pa2753/synvid; O=$SV/t2m2/crops; D=/scratch/th3482/projects/auto-avsr/deliverables/semg3_phone_v1.8_pa_20261105
mkdir -p $O/logs
PIDS=""
for i in $(seq 0 15); do
  $SV/envs/devcrop/bin/python $D/code/ondevice_reference/devcrop.py --reference $SV/v18check/ref \
    --recipe $D/code/hpc/third_party/server_ref_0298/SERVER_RECIPE.json \
    --task-model $D/code/third_party/usr2_e9e9c94/preprocessing/face_landmarker.task \
    --manifest $SV/t2m2/pool/devcrop_manifest.jsonl --out $O/npz --shard $i --nshards 16 > $O/logs/devcrop_$i.log 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p || echo "devcrop $p exit $?"; done
echo "crops: $(ls $O/npz/*.npz | wc -l) / $(wc -l < $SV/t2m2/pool/devcrop_manifest.jsonl)"
cat $O/logs/devcrop_*.log | grep '^{' > $O/devcrop_log.jsonl; grep -h -i "error\|traceback" $O/logs/*.log | head -20
