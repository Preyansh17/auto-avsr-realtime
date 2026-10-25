#!/bin/bash
# Size check for the Whisper decoder-prefix mismatch (design/inference review 2026-10-20, finding inference-1):
# re-decode surviving best/ checkpoints with the prompt they were trained on, <sot><sot><notimestamps>
# (whisper_eval.py --prefix sot_sot), next to the published <sot><en><transcribe><notimestamps> hyps.
# LIST lines: "<run dir> <root dir> <label dir> <splits, comma-separated>"; array task i takes lines i, i+N, ...
#   sbatch --account=<A> --array=0-5 --export=ALL,LIST=<file>,NTASKS=6 prefix_redecode.sbatch
# Writes <run>/<split>_hyps_sotsot.tsv; with CHECK=1 also <run>/<split>_hyps_recheck.tsv (default prefix, must equal
# the published <split>_hyps.tsv).
#SBATCH --job-name=prefix_redecode
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/asr_aug/logs/%x_%A_%a.out
set -euo pipefail
: "${LIST:?}" "${NTASKS:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/pa2753/auto-avsr-realtime
i=0
while read -r RUN ROOT LD SPLITS; do
  if (( i++ % NTASKS == SLURM_ARRAY_TASK_ID )); then
    for s in ${SPLITS//,/ }; do
      [[ -s "$RUN/${s}_hyps_sotsot.tsv" ]] || $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir "$ROOT" \
        --label-file "$LD/${s}_spm1023.csv" --prefix sot_sot --out "$RUN/${s}_hyps_sotsot.tsv"
      if [[ "${CHECK:-0}" == 1 ]]; then
        $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir "$ROOT" --label-file "$LD/${s}_spm1023.csv" \
          --out "$RUN/${s}_hyps_recheck.tsv"
        if cmp -s "$RUN/${s}_hyps.tsv" "$RUN/${s}_hyps_recheck.tsv"; then echo "recheck identical: $RUN $s"
        else echo "RECHECK_DIFFERS: $RUN $s"; fi
      fi
    done
  fi
done < "$LIST"
echo "TASK_DONE $SLURM_ARRAY_TASK_ID"
