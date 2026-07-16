#!/bin/bash
#SBATCH --job-name=carelesswhisper_finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_tandon_advanced

# CarelessWhisper causal-LoRA finetune on patient data, via their
# training_code/train.py. Needs the train/val/test CSVs from
# make_carelesswhisper_dataset.py `csv` mode (wav_path, tg_path, raw_text).
#
# Upstream quirks this script papers over (checked against their code, not
# their README):
#   - train.py imports `ds_dict_private` (NOT the documented ds_dict.py);
#     we generate ${CARELESS_DIR}/ds_dict_private.py from env vars each run.
#   - logs_root/ckpt_root are hardcoded to their /mlspeech machine; we sed
#     them to ${CW_RUNS} (idempotent).
#   - --no_logger sets logger=False, but their LearningRateMonitor callback
#     is hardcoded on regardless and raises MisconfigurationException with
#     no logger attached -- so we do NOT pass --no_logger, and force their
#     wandb logger offline (WANDB_MODE=offline: no account, no network,
#     writes local files) instead of disabling it.
#
# ALL recipe knobs are set explicitly below -- do not trust upstream CLI
# defaults to match their README examples (they don't: e.g. rank defaults to
# 16 in code, 32 in the README base recipe). Their large-v2 README example
# uses a DIFFERENT recipe (rank=4, batch=4) from the small/base one (rank=32,
# batch=16) -- naively reusing the small recipe on large-v2 produced val WER
# ~2x worse than small itself (71.9% vs 30.5%), and using their own large-v2
# recipe only helped marginally (69.9%) -- see
# results/whisper_streaming_investigation_2026-07-09.md. Model size is not a
# free lever here; more training data helped far more than a bigger base.
#
#   TRAIN_CSV=... VAL_CSV=... TEST_CSV=... \
#   SIZE=small GRAN=15 sbatch slurm/carelesswhisper_finetune.sbatch
#
# GRAN is in 20ms encoder frames: 15 = 300ms chunks (their headline config).
# CORRECTED (was wrong earlier): SIZE=large-v3 DOES work -- "no large-v3"
# only applies to their pre-released zero-shot streaming checkpoints (HF hub
# only has base/small/large-v2). Training from scratch via
# load_streaming_model_for_train downloads from the generic OpenAI _MODELS
# dict, which includes large-v3, and whisper_rt's mel pipeline explicitly
# supports both 80-bin (v1/v2/small/base) and 128-bin (v3) filterbanks,
# threaded dynamically from model.dims.n_mels -- not hardcoded to 80.
# Verified by reading the code (mel_filters() asserts n_mels in {80, 128})
# before spending a GPU-hour confirming it, not by trial and error.
#
# SEED is hardcoded (3407) in their train.py with no CLI override -- we
# sed-patch it (idempotent) so SEED=<n> here actually takes effect, needed
# for multi-seed variance runs (LoRA finetunes on this project's data sizes
# swing double-digit points across seeds, see patient-avsr-wer-variance).
#
# EARLY_STOP=1 adds --early_stop (their EarlyStopping callback: monitors
# val/wer, patience=2, mode=min) -- pair with a higher EPOCHS ceiling since
# the callback, not EPOCHS, now decides when training actually stops.
#
# FULL_FINETUNE=1 patches whisper_module.py (idempotent) to unfreeze every
# param, not just LoRA's, at runtime -- their codebase has NO CLI flag for
# this, LoRA is the only supported mode by construction (the freeze loop is
# unconditional). This is the single biggest lever found so far: full FT
# beat every LoRA config on BOTH small (40.81% test WER) and large-v2
# (39.91%, ties small) -- LoRA rank is a fixed trainable-param budget
# independent of base size, so it was starving large-v2 (1.6B params)
# relative to its own size even though the same rank was fine for small
# (240M). Full-FT large-v2 needs BATCH_SIZE=4 for memory headroom
# (unfreezing 1.6B params costs real AdamW memory) but fits fine.
#
# Full-FT checkpoints are MUCH bigger than LoRA's (weights + full optimizer
# state for every trainable param, ~6GB per checkpoint for large-v2/v3 vs
# ~1GB for LoRA-small): default TOP_K=5 x that size across a couple of
# full-FT large-model runs exhausted the scratch quota mid-training (a real
# large-v3 run OOM'd on disk, not GPU memory -- `OSError: Disk quota
# exceeded` from a tqdm progress-bar write, easy to misread as unrelated).
# Pass TOP_K=1 or 2 for full-FT runs on large-v2/v3, and delete a run's raw
# checkpoint directory under ${CW_RUNS}/ckpts/ once you've extracted the one
# checkpoint you need via convert_carelesswhisper_ckpt.py -- the small
# converted .pt in ckpts_finetuned/ is all that's actually needed afterward.
#
# SPECAUG=1 applies SpecAugment (this repo's asr_baselines/specaugment.py,
# Green-parameterized for dysarthric speech: cut freq masking, blow up time
# masking) to the train split only -- their codebase has no augmentation at
# all. Patches datasets_classes.py + whisper_module.py's get_dataset
# (idempotent). Motivation: full-FT converges in ~4 epochs on ~300 clips
# with train loss near zero (clear memorization) -- augmentation is cheap
# insurance against exactly that.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export CW_ENV="${CW_ENV:-/scratch/${USER}/envs/whisper_rt}"
export CW_PY="${CW_ENV}/bin/python"
export CARELESS_DIR="${CARELESS_DIR:-/scratch/${USER}/third_party/CarelessWhisper-streaming}"
export CW_RUNS="${CW_RUNS:-/scratch/${USER}/experiments/carelesswhisper}"
mkdir -p "/scratch/${USER}/auto-avsr/output" "${CW_RUNS}/logs" "${CW_RUNS}/ckpts"

[[ -n "${TRAIN_CSV:-}" && -n "${VAL_CSV:-}" && -n "${TEST_CSV:-}" ]] || {
  echo "[FATAL] set TRAIN_CSV/VAL_CSV/TEST_CSV (from make_carelesswhisper_dataset.py csv)" >&2; exit 4; }
[[ -x "${CW_PY}" ]] || { echo "[FATAL] no python at ${CW_PY}; set CW_ENV" >&2; exit 4; }
[[ -d "${CARELESS_DIR}/training_code" ]] || { echo "[FATAL] no checkout at ${CARELESS_DIR}" >&2; exit 4; }

# Recipe (their README base/small fixed-chunk example, adapted): every value
# explicit, none inherited from upstream argparse defaults.
export DATASET_NAME="${DATASET_NAME:-PATIENT}"
export SEED="${SEED:-3407}"             # their hardcoded default; override for multi-seed runs
# Their checkpoint dirpath is built from cfg.name (RUN_NAME) but NOT the
# seed -- concurrent seed runs with the same default RUN_NAME would write
# into the identical checkpoint directory and clobber each other's epoch
# files. Seed is in the default so this doesn't need remembering per-call.
export RUN_NAME="${RUN_NAME:-patient_$(basename ${TRAIN_CSV} .csv)_seed${SEED}}"
export SIZE="${SIZE:-small}"            # tiny|base|small|medium|large|large-v2 (no v3!)
export GRAN="${GRAN:-15}"               # encoder frames of 20ms; 15 = 300ms chunk
export EXTRA_GRAN_BLOCKS="${EXTRA_GRAN_BLOCKS:-1}"
export RANK="${RANK:-32}"               # LoRA rank (README recipe; code default is 16)
export BATCH_SIZE="${BATCH_SIZE:-16}"
export EPOCHS="${EPOCHS:-10}"
export LR="${LR:-1e-5}"
export STREAMING_FRACTION="${STREAMING_FRACTION:-0.25}"
export TOP_K="${TOP_K:-5}"
export EARLY_STOP="${EARLY_STOP:-0}"    # 1 = --early_stop (monitors val/wer, patience=2)
export FULL_FINETUNE="${FULL_FINETUNE:-0}"  # 1 = unfreeze all params, not just LoRA (patches vendored code)
export SPECAUG="${SPECAUG:-0}"          # 1 = SpecAugment on train split (patches vendored code)

# Point train.py's hardcoded output roots at scratch (idempotent).
sed -i "s|/mlspeech/data/tomer/streaming_whisper/models/logs|${CW_RUNS}/logs|; \
        s|/mlspeech/data/tomer/streaming_whisper/models/ckpts|${CW_RUNS}/ckpts|" \
  "${CARELESS_DIR}/training_code/train.py"

# SEED = 3407 is a literal, no CLI override. Patch it ONCE to read from an
# env var at runtime instead of sed-substituting the literal value each run
# -- substituting the literal would race if multiple seed runs share this
# checkout concurrently (same class of bug as the MFA model-extraction race
# hit earlier: job A patches SEED=1, job B's sed pattern no longer matches
# the now-changed line and silently no-ops, so both jobs could end up
# training with whatever value lost the race). Reading from env at import
# time is safe under concurrency since the file is written only once, ever.
sed -i 's|^SEED = 3407$|SEED = int(os.environ.get("CARELESSWHISPER_SEED", 3407))|' \
  "${CARELESS_DIR}/training_code/train.py"
grep -q '^SEED = int(os.environ.get("CARELESSWHISPER_SEED", 3407))$' "${CARELESS_DIR}/training_code/train.py" || {
  echo "[FATAL] SEED patch didn't take (train.py's SEED line format changed upstream?)" >&2; exit 4; }
export CARELESSWHISPER_SEED="${SEED}"

# whisper_module.py's LoRAStreamedWhisper hardcodes `p.requires_grad = False`
# for every non-LoRA param -- no CLI flag exists to disable this. Patch it to
# check an env var at runtime instead (idempotent: no-ops if already patched).
FULL_FT_MARKER="CARELESSWHISPER_FULL_FINETUNE"
if ! grep -q "${FULL_FT_MARKER}" "${CARELESS_DIR}/training_code/whisper_module.py"; then
  python3 - "${CARELESS_DIR}/training_code/whisper_module.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = '''        for n, p in self.model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False'''
new = '''        import os as _os_ft
        _full_ft = _os_ft.environ.get("CARELESSWHISPER_FULL_FINETUNE") == "1"
        for n, p in self.model.named_parameters():
            if not _full_ft and "lora" not in n:
                p.requires_grad = False'''
assert old in src, "freeze-loop text not found -- upstream code changed?"
open(path, "w").write(src.replace(old, new, 1))
PYEOF
fi

# SpecAugment: their codebase has none. Copy this repo's implementation into
# the checkout (always -- cheap, keeps it in sync with our source of truth)
# and patch AlignedTextGridDatasetLMDB (the only dataset class our
# --streaming_train path actually uses) to apply it on the train split only,
# gated by CARELESSWHISPER_SPECAUG at runtime.
cp "${PROJECT_ROOT}/asr_baselines/specaugment.py" \
   "${CARELESS_DIR}/training_code/carelesswhisper_specaugment.py"

SPECAUG_MARKER="carelesswhisper_specaugment"
if ! grep -q "${SPECAUG_MARKER}" "${CARELESS_DIR}/training_code/datasets_classes.py"; then
  python3 - "${CARELESS_DIR}/training_code/datasets_classes.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

# 1) import (top of file, right after the whisper_rt.tokenizer import)
marker_import = "from training_code.carelesswhisper_specaugment import build_specaugment\n"
anchor = "import whisper_rt.tokenizer\n"
assert anchor in src, "import anchor not found -- upstream imports changed?"
src = src.replace(anchor, anchor + marker_import, 1)

# 2) constructor: add `augment` param + build self.specaug.
# The trailing-comment-free `split='train'):` signature is unique to
# AlignedTextGridDatasetLMDB -- the other AlignedTextGridDataset class has a
# comment on that line, so this text does not match it.
old_sig = """                 multilingual: bool = False,
                 separator='\\t',
                 split='train'):
        super().__init__()"""
new_sig = """                 multilingual: bool = False,
                 separator='\\t',
                 split='train',
                 augment: bool = False):
        super().__init__()"""
assert old_sig in src, "LMDB dataset __init__ signature not found -- upstream code changed?"
src = src.replace(old_sig, new_sig, 1)

BLANK8 = " " * 8  # the "blank" line here is 8 trailing spaces, not truly empty
old_body = ("        self.n_mels = n_mels\n"
            "        self.multilingual = multilingual\n"
            + BLANK8 + "\n"
            "        # LMDB Setup\n"
            "        self.lmdb_paths = lmdb_paths")
new_body = """        self.n_mels = n_mels
        self.multilingual = multilingual
        self.augment = augment
        self.specaug = build_specaugment() if augment else None

        # LMDB Setup
        self.lmdb_paths = lmdb_paths"""
assert old_body in src, "LMDB dataset __init__ body not found -- upstream code changed?"
src = src.replace(old_body, new_body, 1)

# 3) __getitem__: apply after mel is computed, before intervals are loaded.
old_getitem = ('        # 1. Load Audio\n'
               '        audio = whisper_rt.pad_or_trim(whisper_rt.load_audio(item["wav_path"], sr=self.sr))\n'
               '        mel = self._calc_mel(audio)\n'
               + BLANK8 + '\n'
               '        # 2. Load Intervals (Logic Check: LMDB vs File System)')
new_getitem = """        # 1. Load Audio
        audio = whisper_rt.pad_or_trim(whisper_rt.load_audio(item["wav_path"], sr=self.sr))
        mel = self._calc_mel(audio)
        if self.specaug is not None:
            mel = self.specaug(mel)

        # 2. Load Intervals (Logic Check: LMDB vs File System)"""
assert old_getitem in src, "LMDB dataset __getitem__ mel-calc text not found -- upstream code changed?"
src = src.replace(old_getitem, new_getitem, 1)

open(path, "w").write(src)
PYEOF
fi

# whisper_module.py's get_dataset never passes augment= -- wire it to the
# split (train only) and the runtime env var toggle.
if ! grep -q "CARELESSWHISPER_SPECAUG" "${CARELESS_DIR}/training_code/whisper_module.py"; then
  python3 - "${CARELESS_DIR}/training_code/whisper_module.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = 'return AlignedTextGridDatasetLMDB(ds_path=ds_path, lmdb_paths=self.lmdb_paths, get_streamed_mel=True, gran=self.enc_emb_gran, extra_gran_blocks=self.enc_context, n_mels=self.model.dims.n_mels, multilingual=self.cfg.multilingual)'
new = 'return AlignedTextGridDatasetLMDB(ds_path=ds_path, lmdb_paths=self.lmdb_paths, get_streamed_mel=True, gran=self.enc_emb_gran, extra_gran_blocks=self.enc_context, n_mels=self.model.dims.n_mels, multilingual=self.cfg.multilingual, augment=(split == "train" and __import__("os").environ.get("CARELESSWHISPER_SPECAUG") == "1"))'
assert old in src, "get_dataset LMDB call not found -- upstream code changed?"
open(path, "w").write(src.replace(old, new, 1))
PYEOF
fi

# whisper_rt.audio imports pyaudio unconditionally at module load (live-mic
# path, never used here); train.py has no stub for it, unlike our eval
# script. Drop an equivalent stub into the env (idempotent).
CW_SITE="$("${CW_PY}" -c 'import site; print(site.getsitepackages()[0])')"
if [[ ! -f "${CW_SITE}/pyaudio.py" ]]; then
  cat > "${CW_SITE}/pyaudio.py" <<'EOF'
paInt16 = 8
EOF
fi

# train.py does `from ds_dict_private import ds_paths` -- generate it.
cat > "${CARELESS_DIR}/ds_dict_private.py" <<EOF
# Generated by slurm/carelesswhisper_finetune.sbatch $(date -Iseconds); do not edit.
ds_paths = {
    "${DATASET_NAME}": {
        "train": "${TRAIN_CSV}",
        "val": "${VAL_CSV}",
        "test": "${TEST_CSV}",
    },
}
EOF

cd "${CARELESS_DIR}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CARELESS_DIR}:${PYTHONPATH:-}"
# whisper_rt.audio.load_audio shells out to `ffmpeg` on PATH; we invoke
# ${CW_PY} directly rather than activating the env, so PATH needs it too
# (conda-forge ffmpeg installed into CW_ENV -- see requirements doc).
export PATH="${CW_ENV}/bin:${PATH}"
# train_model() unconditionally attaches a LearningRateMonitor callback,
# which raises MisconfigurationException if the Trainer has no logger
# (logger=False whenever --no_logger is passed) -- so we do NOT pass
# --no_logger, and instead force their hardcoded WandbLogger offline (no
# account, no network, writes local files under ${CW_RUNS}/logs/wandb/).
export WANDB_MODE=offline
[[ "${FULL_FINETUNE}" == "1" ]] && export CARELESSWHISPER_FULL_FINETUNE=1
[[ "${SPECAUG}" == "1" ]] && export CARELESSWHISPER_SPECAUG=1

EARLY_STOP_FLAG=""
[[ "${EARLY_STOP}" == "1" ]] && EARLY_STOP_FLAG="--early_stop"

echo "Starting CarelessWhisper finetune at $(date) on $(hostname):"
echo "  SIZE=${SIZE} GRAN=${GRAN} RANK=${RANK} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} LR=${LR} SEED=${SEED}"
echo "  EARLY_STOP=${EARLY_STOP} FULL_FINETUNE=${FULL_FINETUNE} SPECAUG=${SPECAUG}"
echo "  TRAIN_CSV=${TRAIN_CSV}"

"${CW_PY}" training_code/train.py \
  --lora \
  --streaming_train \
  --simulate_stream \
  --dataset "${DATASET_NAME}" \
  --name "${RUN_NAME}" \
  --size "${SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --rank "${RANK}" \
  --gran "${GRAN}" \
  --extra_gran_blocks "${EXTRA_GRAN_BLOCKS}" \
  --streaming_fraction "${STREAMING_FRACTION}" \
  --top_k "${TOP_K}" \
  --strategy auto \
  ${EARLY_STOP_FLAG}

echo "Finished at $(date). Checkpoints under ${CW_RUNS}/ckpts/"
echo "NOTE: lightning .ckpt != the dims/cfg/state_dict .pt that"
echo "carelesswhisper_streaming_eval.py loads -- convert before eval, and"
echo "assert the conversion is BIT-EXACT (see whisper_streaming_investigation)."
