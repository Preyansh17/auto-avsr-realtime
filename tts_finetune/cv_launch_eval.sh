#!/bin/bash
# Experiment K evaluation chain, run on the login node (light) once all 15 fold adapters exist:
#   plans (hash-checked) -> merged gen plan -> gen array over 4 accounts -> items -> PathBench + neural -> table.
set -euo pipefail
cd /scratch/pa2753/higgs_ft
export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/scratch/pa2753/envs/nemo_asr/bin/python
mkdir -p cvplan exp_K
for k in 0 1 2 3 4; do
  for a in "A_r16_cv$k epoch_12" "B_r64_cv$k epoch_08" "C_r16_lr5e-5_cv$k epoch_12"; do set -- $a
    [ -f runs/$1/$2/adapter_model.safetensors ] || { echo "MISSING runs/$1/$2"; exit 3; }; done
  $PY build_plan_ft.py --out cvplan/plan_cv$k.json --split tts_test --splits-file data/cv/splits_cv$k.json --ref ft_stitch5 --seeds 1 2 3 \
    --adapter A_r16_cv$k=runs/A_r16_cv$k/epoch_12 --adapter B_r64_cv$k=runs/B_r64_cv$k/epoch_08 \
    --adapter C_r16_lr5e-5_cv$k=runs/C_r16_lr5e-5_cv$k/epoch_12 --adapter pretrained_cv$k=none | tail -1
done
python3 - <<'PYEOF'
import json
plans = [json.load(open(f"cvplan/plan_cv{k}.json")) for k in range(5)]
units = [u for p in plans for u in p["units"]]
assert len({u["name"] for u in units}) == len(units), "duplicate unit names across folds"
json.dump({"model_path": plans[0]["model_path"], "units": units}, open("cvplan/plan_all.json", "w"), indent=1)
print(len(units), "units,", sum(len(u["prompts"]) for u in units), "clips -> cvplan/plan_all.json")
PYEOF
P=/scratch/pa2753/higgs_ft/cvplan/plan_all.json; ids=""
for spec in "torch_pr_60_general 0-2" "torch_pr_60_tandon_advanced 3-5" "torch_pr_39_general 6-8" "torch_pr_60_tandon_priority 9-11"; do set -- $spec
  j=$(sbatch --parsable --account=$1 -p l40s_public --array=$2 --export=ALL,PLAN=$P,NTASKS=12 final_gen.sbatch); echo "gen $1 [$2] -> $j"; ids="$ids:$j"; done
ids=${ids#:}
WR="export PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /scratch/pa2753/higgs_ft; $PY"
JI=$(sbatch --parsable --account=torch_pr_60_general -p cpu_short --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=00:15:00 --job-name=msK_items -o logs/msK_items_%j.out --dependency=afterany:$ids --wrap="$WR multi_eval.py items --exp K --chunks 6 --allow-incomplete"); echo items=$JI
JP=$(sbatch --parsable --account=torch_pr_60_tandon_advanced -p l40s_public --array=0-5 --dependency=afterany:$JI --export=ALL,FINAL=/scratch/pa2753/higgs_ft/exp_K final_score.sbatch); echo pb=$JP
JN=$(sbatch --parsable --account=torch_pr_39_general -p l40s_public --array=0-7 --dependency=afterany:$JI --export=ALL,NAMES_FILE=/scratch/pa2753/higgs_ft/exp_K/neural_todo.txt,NTASKS=8 neural_multi.sbatch); echo neural=$JN
JT=$(sbatch --parsable --account=torch_pr_60_general -p cpu_short --gres=gpu:0 --cpus-per-task=4 --mem=24G --time=00:20:00 --job-name=msK_table -o logs/msK_table_%j.out --dependency=afterany:$JP:$JN --wrap="$WR multi_eval.py table --exp K"); echo table=$JT
echo "$ids $JI $JP $JN $JT" > exp_K_jobs.txt
