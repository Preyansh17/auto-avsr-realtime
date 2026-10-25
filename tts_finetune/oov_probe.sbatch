#!/bin/bash
# Generalization probe on the finetuned adapter: generate the 50 out-of-set sentences with the adapter AND with the
# pretrained model (same reference, same seeds), then score them. Environment mirrors higgs_gen/gen.sbatch.
# The account/partition are NOT set here; pass them (and spread parallel tasks over the accounts, see below):
#
#   python oov_probe.py build                                     # once, CPU, seconds (login node is fine)
#   python oov_probe.py plan --adapter runs/<run>/epoch_<NN> --ref ft_stitch5   # writes data/plan_oov.json (4 units)
#
#   # one job does all 4 units (~10-15 min on an L40S):
#   sbatch --account=<acct> [-p <partition>] --export=ALL,NTASKS=1 oov_probe.sbatch
#   # or split the units over accounts that start fastest; each task = one chunk of BOTH arms (gen_ft sorts by adapter):
#   sbatch --account=<acct1> --export=ALL,NTASKS=2,TASK=0 oov_probe.sbatch
#   sbatch --account=<acct2> --export=ALL,NTASKS=2,TASK=1 oov_probe.sbatch
#   # then score once every task has finished (PER: any GPU, or a CPU node, ~1-2 s/clip on CPU):
#   sbatch --account=<acct> --gres=gpu:0 -p <cpu partition> --time=00:20:00 --export=ALL,STAGE=score oov_probe.sbatch
#   # (or --dependency=afterok:<id1>:<id2> on the score job)
#
# Env knobs: PLAN (default data/plan_oov.json), NTASKS (default 1), TASK (default $SLURM_ARRAY_TASK_ID or 0),
#            STAGE = gen | score | both (default gen; "both" needs NTASKS=1), EXTRA = extra gen_ft.py args,
#            SCORE_EXTRA = extra `oov_probe.py score` args (e.g. "--per off", "--out-dir <dir>").
#SBATCH --job-name=higgs_oov_probe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%j.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/scratch/pa2753/envs/nemo_asr/bin/python
PLAN=${PLAN:-/scratch/pa2753/higgs_ft/data/plan_oov.json}
STAGE=${STAGE:-gen}
NTASKS=${NTASKS:-1}
TASK=${TASK:-${SLURM_ARRAY_TASK_ID:-0}}
case "$STAGE" in gen|score|both) ;; *) echo "STAGE must be gen|score|both, got '$STAGE'" >&2; exit 2;; esac
if [ "$STAGE" = both ] && [ "$NTASKS" != 1 ]; then echo "STAGE=both needs NTASKS=1 (scoring needs every unit finished)" >&2; exit 2; fi
cd /scratch/pa2753/higgs_ft
echo "host=$(hostname) job=${SLURM_JOB_ID:-} stage=$STAGE plan=$PLAN task=$TASK/$NTASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU on this node"

if [ "$STAGE" = gen ] || [ "$STAGE" = both ]; then
  $PY gen_ft.py --plan "$PLAN" --task "$TASK" --num-tasks "$NTASKS" ${EXTRA:-}
fi
if [ "$STAGE" = score ] || [ "$STAGE" = both ]; then
  # PathBench needs its own env (pb_site + pathbench on PYTHONPATH), not the transformers 5.12.1 overlay used for Higgs
  ( unset PYTHONPATH; source /scratch/pa2753/pb_eval/pb_env.sh; export HF_HUB_OFFLINE=1
    "$PB_PY" oov_probe.py score --plan "$PLAN" ${SCORE_EXTRA:-} )
fi
echo DONE
