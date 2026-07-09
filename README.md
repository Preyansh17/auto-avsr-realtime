# auto-avsr-realtime: Streaming Audio-Visual Speech Recognition

Real-time (streaming) AVSR fork of [auto_avsr](https://github.com/mpc001/auto_avsr), following the PyTorch blog
[Real-time AV-ASR](https://pytorch.org/blog/real-time-speech-rec/): an **Emformer RNN-T transducer**
decoded chunk-by-chunk with carried state, instead of the offline Conformer + CTC/attention
beam search (which remains in the original `auto-avsr` repo).

Input is **file-based streaming**: existing videos are consumed segment-by-segment as if they
arrived live, with incremental transcript output. There is no microphone/camera capture.

Built on the [torchaudio examples/avsr recipe](https://github.com/pytorch/audio/tree/main/examples/avsr)
and the device_avsr tutorial assets.

## What's here

| Path | Purpose |
| --- | --- |
| `online_avsr/` | Core package: model factories, Lightning module, streaming pipeline, transforms, patient dataset/datamodule |
| `demo_realtime.py` | Stream a video file through the model with incremental transcript + RTF/latency stats |
| `train.py` / `eval.py` | Fine-tune (or train from scratch) and evaluate on patient data |
| `asr_baselines/` | Audio-only ASR baselines (Whisper, NVIDIA Nemotron streaming) the AV model is benchmarked against — see `asr_baselines/README.md` |
| `results/` | Dated investigation write-ups (weekly findings, bug postmortems, open items) — the source of truth for "what's the current best number and why" |
| `scripts/download_assets.py` | Fetch the pretrained streaming model + SentencePiece vocab |
| `scripts/bootstrap_from_jit.py` | Convert the pretrained TorchScript model into a fine-tunable eager checkpoint |
| `scripts/regenerate_patient_labels.py` | Re-tokenize old label CSVs (unigram5000 → spm_unigram_1023) |
| `scripts/merge_lora_ckpt.py` | Merge any saved LoRA epoch checkpoint into plain weights for eval, without retraining |
| `scripts/read_eval_summary.py` | Read a field (or whole dict) out of an `eval.py` run's `summary.json`, by glob — used by sweep scripts to avoid fragile inline shell quoting |
| `slurm/` | NYU HPC templates (singularity + conda) for fine-tune / scratch / eval / label regen / checkpoint selection |
| `preparation/` | Face & mouth-ROI detectors (mediapipe, retinaface, and torchaudio's face-crop variant) |

## Current best results

Streaming WER on patient data, honest (train/val-selection/held-out-test split, no
double-dipping between checkpoint selection and reporting):

| Model | Legal-only | Merged |
| --- | --- | --- |
| **Whisper large-v3 + SpecAugment, full finetune** | **5.5%** | **2.6%** |
| Whisper medium + SpecAugment, full finetune | 9.8% | 2.6% |
| AV Emformer, audio-only, LoRA (streaming-selected) | 26.8% | — |
| Nemotron streaming, full finetune | 33.3% | 31.7% |
| AV Emformer, audio-visual, LoRA (streaming-selected) | 36.6% | — |

Video does not currently help: audio-only configs beat their audio-visual counterparts
across every architecture tested. See `results/week_results_2026-06-23_2026-07-01.md`
(§27-31) and `results/week_results_2026-07-02_2026-07-06.md` for the full investigation,
including two real methodology bugs found and fixed along the way — a WER metric
mismatch between architectures (§27) and a checkpoint-selection bias in Whisper's
default recipe (§28-30) that briefly inflated its headline number before an honest
three-way split corrected it. `results/whisper_streaming_investigation_2026-07-09.md`
documents an unresolved attempt at real-time streaming Whisper (blocked, not just
untried — see that file for the current state and options).

## Two architectures

| | `device` (pretrained) | `recipe` |
| --- | --- | --- |
| Video frontend | Linear(44×44 → 512) on face crops | Conv3d+ResNet18 on 88×88 mouth ROIs |
| Emformer | 12 layers, dim 256, ffn 1024 | 20 layers, dim 128, ffn 2048 |
| Streaming cadence | segment 32 + right-context 4 (1.44 s) | segment 64, rc 0 (2.56 s) by default |
| Pretrained weights | yes (`bootstrap_from_jit.py`, bit-exact) | no |

The published pretrained model is the blog's "Small" device configuration. The bootstrap maps
**100% of parameters with zero numerical difference**, so fine-tuning starts from the real
pretrained model. Segment/right-context lengths are decode-time settings — weight shapes don't
depend on them.

## Setup (local, macOS/Linux)

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and all pins come from `pyproject.toml`
(torch/torchaudio **2.6.0** — the RNN-T streaming APIs are deprecated from torchaudio 2.8, do
not upgrade casually; mediapipe ≤ 0.10.21 for the legacy `mp.solutions` API).

```bash
uv venv --python 3.12 && uv sync
source .venv/bin/activate
python scripts/download_assets.py          # pretrained JIT model -> cpts/, spm -> spm/
python scripts/bootstrap_from_jit.py       # eager fine-tunable checkpoint + parity report
python -m tests.test_smoke                 # CPU smoke tests
```

## Streaming demo

```bash
# Pretrained model, any talking-head video (mediapipe face crops):
python demo_realtime.py --video clip.mp4 --jit-model cpts/device_avsr_model.pt --carry-state

# Same weights, eager checkpoint (canonical windowing with true lookahead):
python demo_realtime.py --video clip.mp4 --checkpoint cpts/online_avsr_bootstrap.ckpt

# Fine-tuned checkpoint on an already-cropped patient mouth-ROI mp4 (sibling .wav picked up):
python demo_realtime.py --video roi_clip.mp4 --checkpoint exp/run/last.ckpt --preprocess roi

# Pace output like a live source and log per-chunk records:
python demo_realtime.py --video clip.mp4 --simulate-realtime --output-jsonl out.jsonl
```

`--preprocess`: `face` (mediapipe detect → align → face crop; what the pretrained model expects),
`mouth` (auto-avsr mouth-ROI detection), `roi` (input already cropped), `none` (raw resize, smoke
tests). Defaults follow the model. Reference example (12 s public-domain clip, M-series CPU):
overall RTF ≈ 0.2, algorithmic latency = (segment + right context)/25 fps = 1.44 s.

## Patient fine-tuning workflow

1. **Re-tokenize labels** (old CSVs carry uppercase unigram5000 ids; the streaming model uses the
   lowercase 1023-piece vocab):
   ```bash
   python scripts/regenerate_patient_labels.py --old-csv labels/train.csv --preview 5  # eyeball
   python scripts/regenerate_patient_labels.py --old-csv labels/train.csv labels/val.csv
   ```
2. **Fine-tune from the pretrained bootstrap** (device architecture, 44×44 frames):
   ```bash
   python train.py --model-source bootstrap \
     --root-dir /path/to/patient_data \
     --train-file labels/train_spm1023.csv --val-file labels/val_spm1023.csv \
     --precision bf16-mixed --batch-size 2 --accumulate-grad-batches 4
   ```
   On the cluster: `sbatch slurm/train_realtime_finetune.sbatch` (paths/epochs via env vars).

   **LoRA variant** — freeze the pretrained weights and train low-rank adapters only
   (~2.4% of parameters with the defaults; good for small patient sets):
   ```bash
   python train.py --model-source bootstrap --lora \
     --lora-r 8 --lora-alpha 16 --lora-scopes encoder predictor joiner fusion \
     --root-dir /path/to/patient_data \
     --train-file labels/train_spm1023.csv --val-file labels/val_spm1023.csv
   ```
   Cluster: `sbatch slurm/train_realtime_lora.sbatch`. The run directory gets
   `model_lora_merged.pth` (adapters folded back into plain weights) which eval.py and
   demo_realtime.py consume directly; `last.ckpt` keeps the LoRA form for resuming.
   Scopes: `encoder` (Emformer), `predictor`, `joiner`, `fusion`, `video_frontend`, or `all`.
3. **Evaluate** (streaming WER + RTF, or offline utterance mode):
   ```bash
   python eval.py --checkpoint exp/run/last.ckpt --mode streaming \
     --root-dir /path/to/patient_data --test-file labels/test_spm1023.csv --preprocess roi
   ```
   The summary reports both `avg_wer` (macro-average of per-clip WER, kept for back-compat)
   and `corpus_wer` (pooled: total edits / total ref words — the standard WER convention,
   also what `asr_baselines/metrics.py` uses for Whisper/Nemotron). **Use `corpus_wer`** for
   any comparison against those other models; the two are not the same number and differ by
   a few points on this data.
4. **Select the best checkpoint properly** (optional, but strongly recommended). The default
   recipe merges whatever the *last* epoch produced — no selection at all. Two selection
   metrics were tried and found **not to track real streaming WER** on this recipe: `val_loss`
   anti-correlates with WER, and NeMo/HF-style per-epoch greedy-utterance `val_wer` doesn't
   track streaming-beam WER either (see `results/week_results_2026-06-23_2026-07-01.md` §4-5).
   The fix that actually works: decode every candidate epoch in **real streaming mode** against
   a selection set, pick the best, report on a disjoint held-out set:
   ```bash
   # Train with AVG_LAST_N>0 so Lightning keeps the last N epoch checkpoints on disk:
   AVG_LAST_N=20 sbatch slurm/train_realtime_lora.sbatch
   # Then sweep them in real streaming mode and report on a held-out test set:
   RUN_DIR=<exp-dir>/<run> SELECT_FILE=val_spm1023.csv TEST_FILE=test_spm1023.csv \
     sbatch slurm/select_best_streaming_epoch.sbatch
   ```
   This found a large, consistent win over merge-last on every seed tested — see
   `results/week_results_2026-06-23_2026-07-01.md` §31.

Notes:
- Fine-tunes from the bootstrap **must** keep `spm/spm_unigram_1023.model` (enforced; sha256 is
  recorded in the run manifest). `--generate-sp-model` is for from-scratch runs only.
- The pretrained frontend saw *face* crops; patient data is *mouth ROIs* — a known distribution
  shift that fine-tuning absorbs. Do not re-run face detection on already-cropped patient clips;
  use `--preprocess roi` end to end.
- RNNT loss memory grows with sequence length × target length; patient CSVs are 24 s-segmented
  and `--max-frames 600` drops outliers. Use batch 1–2 + `--accumulate-grad-batches`.
- This recipe has **large run-to-run WER variance when unseeded** (~24pp observed across
  otherwise-identical runs) — always pass `--seed`/`SEED` and compare medians over ≥3 seeds,
  never trust a single run. See `results/week_results_2026-06-23_2026-07-01.md` §10 for the
  investigation that found this.

## Cluster environment (NYU HPC)

The sbatch templates expect a conda env `avsr_realtime` inside the singularity overlay.
Create it once with the overlay mounted **read-write** (the run templates mount it `:ro`):

```bash
singularity exec --overlay /scratch/$USER/avsr/overlay-15GB-500K.ext3:rw \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash
# inside:
source /ext3/env.sh
conda create -y -n avsr_realtime python=3.12 && conda activate avsr_realtime
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install "pytorch-lightning>=2.4,<3" sentencepiece "mediapipe<=0.10.21" opencv-python scikit-image av soundfile
```

### Patient data layout (matches the offline runs)

The training/eval templates resolve patient data exactly like the offline
`train_patient_av_lora_legal298_*` scripts, via `slurm/_patient_data.sh`:

- `RUN_DATA_MODE=legal_only` (default) → `patient_legal298_crops_unseen`
- `RUN_DATA_MODE=legacy_only` → `patient_25p_crops_unseen`
- `RUN_DATA_MODE=merged` (or `MERGE_WITH_PATIENT_UNSEEN=1`) → builds the legal298+unseen
  merge (per-source `dataset_name` rewrite + symlinks), same as the offline merge.

It reads the same `patient_retinaface_{train,val}_transcript_lengths_seg24s.csv` files and,
before training, re-tokenizes them to the 1023-piece vocab (`*_spm1023.csv`).

All templates take `PROJECT_ROOT`, `ROOT_DIR`, `RUN_DATA_MODE`, `LORA_*`, `MAX_STEPS`,
`MAX_EPOCHS`, etc. as env-var overrides, e.g. `MERGE_WITH_PATIENT_UNSEEN=1 sbatch slurm/train_realtime_lora.sbatch`.

### What carries over from the offline LoRA runs, and what doesn't

| Offline arg | Streaming equivalent |
| --- | --- |
| `lora.r=8 alpha=16 dropout=0.05` | identical (`--lora-r/--lora-alpha/--lora-dropout`) |
| `lora.scopes=[encoder,aux_encoder,decoder]` ("all") | `--lora-scopes all` (encoder/predictor/joiner/fusion/video_frontend) |
| `max_steps=2850`, `max_epochs=10000` | identical (`--max-steps`, `--epochs`) |
| `pretrained_model_path=…Conformer.pth` | `cpts/online_avsr_bootstrap.ckpt` (device_avsr) |
| `vocab_file=…sentences.txt` (closed-vocab decode) | **none** — RNN-T decodes open-vocabulary |
| `ctc_weight=0.1`, `beam_size=40`, `pre_beam_ratio` | **none in training** — RNN-T has no CTC; beam is decode-only (`eval.py --beam-width`) |
| `data.modality=audiovisual` | AV only (the streaming model is audio-visual) |

## How streaming works

Frames are consumed in fixed windows: `lookback` past frames (frontend receptive field, trimmed
from the features), the new `segment`, and `right-context` lookahead frames. `Emformer.infer`
requires **exactly** segment + right-context fused frames per call; `RNNTBeamSearch.infer` carries
encoder state and the beam hypothesis across calls (`--no-carry-state` reproduces the tutorial's
per-chunk reset instead). Algorithmic latency = (segment + rc)/25 fps; wall-clock per-chunk RTF is
reported by the demo and eval.

## Provenance / licenses

- Recipe code ported from [pytorch/audio examples/avsr](https://github.com/pytorch/audio/tree/main/examples/avsr) (BSD-2-Clause).
- `preparation/detectors/mediapipe_face/` is torchaudio's data_prep mediapipe detector (face crops);
  the original auto_avsr detectors (mouth ROIs, Apache 2.0 headers) are kept alongside.
- Pretrained weights: torchaudio `tutorial-assets/device_avsr_model.pt` (BSD-2-Clause), trained on
  LRS3 + VoxCeleb2 + AVSpeech per Ma et al., "Auto-AVSR" (ICASSP 2023) and the
  real-time AV-ASR blog post.
