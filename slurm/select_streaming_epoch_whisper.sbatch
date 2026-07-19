#!/bin/bash
#SBATCH --job-name=select_streaming_epoch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Streaming-aware checkpoint selection for WHISPER.
#
# Every Whisper result in this investigation selects its checkpoint by OFFLINE
# val WER (Seq2SeqTrainingArguments metric_for_best_model="wer"). Offline and
# streaming WER are known not to rank checkpoints identically here (the
# legacy-domain seed flip earlier in the investigation is the clearest case:
# the best offline seed was the worst streaming seed), so offline-val
# selection may be systematically mispicking for the streaming deployment.
#
# This decodes every RETAINED epoch checkpoint of a run in actual streaming
# mode (0.6s segments) against the VAL split, so the best-by-streaming epoch
# can be chosen honestly, and also reports each one on the disjoint TEST split
# so the selected checkpoint's test number can be quoted without re-running.
# SELECTION USES VAL ONLY -- the test numbers are printed for reporting the
# selected checkpoint, not for choosing it.
#
# Note save_total_limit=3, so the candidate pool is only the 3 best-by-OFFLINE
# checkpoints plus best/ -- already biased toward offline. If a different
# epoch wins here, a retrain with a larger save_total_limit is warranted to
# search a real pool.
#
# (slurm/select_best_streaming_epoch.sbatch does this for the AV Emformer:
# Lightning .ckpt + LoRA merge + singularity + eval.py. None of that applies
# to the Whisper HF-Trainer pipeline, hence this separate script.)
#
#   RUN=whisper_largev3_specaug_splitv1_merged_wdwarmup_seed1 \
#     sbatch slurm/select_streaming_epoch_whisper.sbatch

export TMPDIR=/scratch/pa2753/.tmp
export HF_HOME=/scratch/pa2753/hf_cache
export PYTHONNOUSERSITE=1
EXP=/scratch/pa2753/experiments/whisper_asr
RUN="${RUN:?set RUN=<experiment dir name under $EXP>}"
RUN_DIR="$EXP/$RUN"
[[ -d "$RUN_DIR" ]] || { echo "[FATAL] no $RUN_DIR" >&2; exit 3; }

SPLIT_ROOT=/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1
LABELS=/scratch/pa2753/avsr_realtime/labels/splitv1/merged
SPM=/home/pa2753/auto-avsr-realtime/spm/spm_unigram_1023.model
VAL_MANIFEST="$RUN_DIR/val_manifest.tsv"
# The run's own val_hyps.tsv is the TEST-set output (the finetune sbatch runs
# whisper_eval on TEST_FILE), so it serves as the test manifest.
TEST_MANIFEST="$RUN_DIR/val_hyps.tsv"

echo "########## build VAL manifest (path/ref) for merged val split"
SPLIT_ROOT="$SPLIT_ROOT" LABELS="$LABELS" SPM="$SPM" VAL_MANIFEST="$VAL_MANIFEST" \
/scratch/pa2753/envs/whisper_asr/bin/python - <<'PYEOF'
import os, sys
sys.path.insert(0, "/home/pa2753/auto-avsr-realtime")
from asr_baselines.patient_audio import load_audio_examples

ex = load_audio_examples(os.environ["SPLIT_ROOT"],
                         os.path.join(os.environ["LABELS"], "val_spm1023.csv"),
                         os.environ["SPM"])
out = os.environ["VAL_MANIFEST"]
with open(out, "w", encoding="utf-8") as f:
    f.write("path\tref\n")
    for e in ex:
        f.write(f"{e.path}\t{e.text}\n")
print(f"wrote {out} ({len(ex)} val clips)")
PYEOF

export PATH=/scratch/pa2753/envs/simulstream/bin:$PATH
export LD_LIBRARY_PATH=/scratch/pa2753/envs/simulstream/lib:${LD_LIBRARY_PATH:-}

shopt -s nullglob
for CKPT_DIR in "$RUN_DIR"/checkpoint-* "$RUN_DIR"/best; do
  [[ -d "$CKPT_DIR" ]] || continue
  NAME=$(basename "$CKPT_DIR")
  OUT_PT="$RUN_DIR/openai_format/sel-${NAME}.pt"
  mkdir -p "$RUN_DIR/openai_format"

  echo "########## convert ${NAME} (bit-exact check)"
  CKPT_DIR="$CKPT_DIR" OUT_PT="$OUT_PT" /scratch/pa2753/envs/whisper_asr/bin/python - <<'PYEOF'
import sys, os, torch
sys.path.insert(0, "/scratch/pa2753/third_party")
from convert_hf_to_openai import convert_tfms_to_openai_whisper, reverse_rename_keys
from transformers import WhisperForConditionalGeneration

hf_ckpt, out_pt = os.environ["CKPT_DIR"], os.environ["OUT_PT"]
convert_tfms_to_openai_whisper(hf_ckpt, out_pt)

hf = WhisperForConditionalGeneration.from_pretrained(hf_ckpt)
sd_hf = {k: v.cpu() for k, v in hf.state_dict().items()}
sd_hf = reverse_rename_keys(sd_hf)
del sd_hf["proj_out.weight"]
sd_hf = {k.replace("model.", "", 1): v for k, v in sd_hf.items()}
sd_new = torch.load(out_pt, map_location="cpu")["model_state_dict"]
key_mismatch = set(sd_hf) != set(sd_new)
worst = max((sd_hf[k] - sd_new[k]).abs().max().item() for k in sd_new if k in sd_hf) if not key_mismatch else float("inf")
dtypes = {str(v.dtype) for v in sd_new.values()}
ok = (not key_mismatch) and worst == 0.0 and dtypes == {"torch.float32"}
print(f"{os.path.basename(hf_ckpt)}: worst_diff={worst} dtypes={dtypes} bit_exact={ok}")
if not ok:
    os.path.exists(out_pt) and os.remove(out_pt)
    sys.exit(1)
PYEOF

  cd /scratch/pa2753/wse_pkg
  echo "########## SELECT(val) ${NAME}"
  python -m asr_baselines.whisper_streaming_eval \
    --model-path "$OUT_PT" --manifest-tsv "$VAL_MANIFEST" \
    --simulstreaming-dir /scratch/pa2753/third_party/SimulStreaming \
    --segment-length 0.6 --out "$RUN_DIR/sel_${NAME}_val_hyps.tsv"
  echo "########## REPORT(test) ${NAME}"
  python -m asr_baselines.whisper_streaming_eval \
    --model-path "$OUT_PT" --manifest-tsv "$TEST_MANIFEST" \
    --simulstreaming-dir /scratch/pa2753/third_party/SimulStreaming \
    --segment-length 0.6 --out "$RUN_DIR/sel_${NAME}_test_hyps.tsv"
  cd - >/dev/null
done

echo "Finished at $(date)"
