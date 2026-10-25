#!/bin/bash
# TitaNet + WavLM embeddings (tts_eval/v2/neural_metrics_v2.py, exactly as neural_shard.sbatch) for MANY generated runs in
# one job. Lines of $NAMES_FILE are "NAME|RUN_DIR" (RUN_DIR under /scratch/pa2753/gen_runs, output tts_eval/out_v2/NAME).
# Array task i of NTASKS handles lines i, i+NTASKS, ...; runs whose out dir already holds neural.shard*.csv are skipped.
#   sbatch --account=<A> -p <P> --array=0-3 --export=ALL,NAMES_FILE=<exp_dir>/neural_todo.txt,NTASKS=8 neural_multi.sbatch
#SBATCH --job-name=ft_neural_multi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
set -uo pipefail
: "${NAMES_FILE:?set NAMES_FILE}" "${NTASKS:?set NTASKS}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} task=$TASK/$NTASKS file=$NAMES_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
i=0; fails=0
while IFS='|' read -r NAME RUN_DIR; do
  [ -z "$NAME" ] && continue
  if [ $((i % NTASKS)) -eq "$TASK" ]; then
    if ls /scratch/pa2753/tts_eval/out_v2/$NAME/neural.shard*.csv >/dev/null 2>&1; then echo "skip $NAME (done)"; else
      echo "== $NAME ($RUN_DIR)"
      ( export NAME RUN_DIR RUNS_ROOT=/scratch/pa2753/gen_runs NSHARDS=1
        source /scratch/pa2753/tts_eval/v2/run_env.sh
        "$PY" neural_metrics_v2.py "${ARGS[@]}" --out-dir "$OUT" --shard 0 --num-shards 1 ) || { echo "FAILED $NAME"; fails=$((fails+1)); }
    fi
  fi
  i=$((i+1))
done < "$NAMES_FILE"
echo "TASK_DONE fails=$fails"
[ "$fails" -eq 0 ]
