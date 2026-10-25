#!/bin/bash
# Experiment R chain (reference study, fold 0): plans -> gen array over 4 accounts -> items -> PathBench + neural -> table.
set -euo pipefail
cd /scratch/pa2753/higgs_ft
export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/scratch/pa2753/envs/nemo_asr/bin/python
mkdir -p cvplan exp_R
for r in stitch5 lowpitch5 rs_rand1 rs_rand3 rs_rand5a rs_rand5b rs_vocab5; do
  $PY build_plan_ft.py --out cvplan/plan_R_$r.json --split tts_test --splits-file data/cv/splits_cv0.json \
    --ref $r --ref-json refs/refs_study.json --seeds 1 2 3 \
    --adapter B_r64_cv0=runs/B_r64_cv0/epoch_08 --adapter pretrained_cv0=none | tail -1
done
python3 - <<'PYEOF'
import json
plans = [json.load(open(f"cvplan/plan_R_{r}.json")) for r in "stitch5 lowpitch5 rs_rand1 rs_rand3 rs_rand5a rs_rand5b rs_vocab5".split()]
units = [u for p in plans for u in p["units"]]
assert len({u["name"] for u in units}) == len(units)
json.dump({"model_path": plans[0]["model_path"], "units": units}, open("cvplan/plan_R_all.json", "w"), indent=1)
print(len(units), "units,", sum(len(u["prompts"]) for u in units), "clips")
PYEOF
P=/scratch/pa2753/higgs_ft/cvplan/plan_R_all.json; ids=""
for spec in "torch_pr_60_general 0-2" "torch_pr_60_tandon_advanced 3-5" "torch_pr_39_general 6-8" "torch_pr_60_tandon_priority 9-11"; do set -- $spec
  j=$(sbatch --parsable --account=$1 -p l40s_public --array=$2 --export=ALL,PLAN=$P,NTASKS=12 final_gen.sbatch); echo "gen $1 [$2] -> $j"; ids="$ids:$j"; done
ids=${ids#:}
WR="export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /scratch/pa2753/higgs_ft; $PY"
JI=$(sbatch --parsable --account=torch_pr_60_general -p cpu_short --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=00:15:00 --job-name=msR_items -o logs/msR_items_%j.out --dependency=afterany:$ids --wrap="$WR ref_study.py items --chunks 3 --allow-incomplete"); echo items=$JI
JP=$(sbatch --parsable --account=torch_pr_60_tandon_advanced -p l40s_public --array=0-2 --dependency=afterany:$JI --export=ALL,FINAL=/scratch/pa2753/higgs_ft/exp_R final_score.sbatch); echo pb=$JP
JN=$(sbatch --parsable --account=torch_pr_39_general -p l40s_public --array=0-7 --dependency=afterany:$JI --export=ALL,NAMES_FILE=/scratch/pa2753/higgs_ft/exp_R/neural_todo.txt,NTASKS=8 neural_multi.sbatch); echo neural=$JN
JT=$(sbatch --parsable --account=torch_pr_60_general -p cpu_short --gres=gpu:0 --cpus-per-task=4 --mem=24G --time=00:20:00 --job-name=msR_table -o logs/msR_table_%j.out --dependency=afterany:$JP:$JN --wrap="$WR ref_study.py table"); echo table=$JT
echo "$ids $JI $JP $JN $JT" > exp_R_jobs.txt
