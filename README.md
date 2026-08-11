# auto-avsr-realtime

Streaming audio-visual speech recognition for dysarthric patient speech — an **Emformer RNN-T**
transducer decoded chunk-by-chunk with carried state, forked from
[auto_avsr](https://github.com/mpc001/auto_avsr) and following the PyTorch
[Real-time AV-ASR](https://pytorch.org/blog/real-time-speech-rec/) blog.

Streaming is **file-based**: clips are consumed segment-by-segment as if arriving live, with
incremental transcript output. There is no microphone/camera capture — this keeps evaluation
reproducible and latency measurable against ground truth.

Audio-only baselines (Whisper, NVIDIA Nemotron) live in `asr_baselines/` and are benchmarked
against the AV model to answer the project's driving question: **does the video stream help?**

## Results

Streaming decode on the audited three-way patient split, 3 seeds per configuration:

| System | Merged WER | Legal WER (cross-domain) | TTFT | Word lag | RTF |
| --- | --- | --- | --- | --- | --- |
| **Whisper large-v3 + SpecAugment**, full FT, 1.2 s chunks | **8.97%** | **6.56%** | 1.76 s | 1.61 s | 0.170 |
| **Nemotron 0.6B**, 210 epochs + speed perturbation | 10.76% | **6.56%** | **0.80 s** | **0.70 s** | **0.02–0.03** |
| AV Emformer, audio-only (streaming-selected) | — | 26.8% | — | — | ~0.2 |
| AV Emformer, audio-visual (streaming-selected) | — | 36.6% | — | — | ~0.2 |

Whisper is the accuracy choice; Nemotron is ~2× lower latency and ~7× cheaper for ~1.8pp of merged
WER. They tie on the like-for-like cross-domain test. **Audio-only beats audio-visual in every
architecture tested** — video does not currently earn its cost.

Latency is measured in simulated real time, anchored to speech onset, against forced-alignment
ground truth. Full analysis: **[`results/project_report_2026-08-10.md`](results/project_report_2026-08-10.md)**.
Dated investigation logs (weekly findings, sweep tables, postmortems) are in
[`docs/investigations/`](docs/investigations/).

## Layout

| Path | Purpose |
| --- | --- |
| `online_avsr/` | Core package: model factories, Lightning module, streaming pipeline, transforms, patient dataset |
| `train.py` / `eval.py` | Fine-tune (or train from scratch) and evaluate |
| `demo_realtime.py` | Stream a video file with incremental transcript + RTF/latency stats |
| `asr_baselines/` | Audio-only baselines (Whisper, Nemotron) — see `asr_baselines/README.md` |
| `scripts/` | Asset download, JIT→eager bootstrap, label re-tokenization, LoRA merge, latency benchmark |
| `slurm/` | NYU HPC templates (singularity + conda) |
| `preparation/` | Face and mouth-ROI detectors |
| `docs/investigations/` | Dated write-ups — the record of how each number was arrived at |

## Two architectures

| | `device` (pretrained) | `recipe` |
| --- | --- | --- |
| Video frontend | Linear(44×44 → 512) on face crops | Conv3d+ResNet18 on 88×88 mouth ROIs |
| Emformer | 12 layers, dim 256, ffn 1024 | 20 layers, dim 128, ffn 2048 |
| Streaming cadence | segment 32 + right-context 4 (1.44 s) | segment 64, rc 0 (2.56 s) |
| Pretrained weights | yes (bit-exact bootstrap) | no |

The bootstrap maps **100% of parameters with zero numerical difference**, so fine-tuning starts
from the real pretrained model. Segment and right-context are decode-time settings — weight shapes
do not depend on them.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Pins come from `pyproject.toml` — torch/torchaudio
**2.6.0** (RNN-T streaming APIs are deprecated from 2.8, do not upgrade casually) and
mediapipe ≤ 0.10.21 for the legacy `mp.solutions` API.

```bash
uv venv --python 3.12 && uv sync
source .venv/bin/activate
python scripts/download_assets.py     # pretrained JIT model -> cpts/, spm -> spm/
python scripts/bootstrap_from_jit.py  # eager fine-tunable checkpoint + parity report
python -m tests.test_smoke
```

## Streaming demo

```bash
python demo_realtime.py --video clip.mp4 --jit-model cpts/device_avsr_model.pt --carry-state
python demo_realtime.py --video roi_clip.mp4 --checkpoint exp/run/last.ckpt --preprocess roi
python demo_realtime.py --video clip.mp4 --simulate-realtime --output-jsonl out.jsonl
```

`--preprocess`: `face` (detect → align → crop; what the pretrained model expects), `mouth`
(auto-avsr mouth-ROI detection), `roi` (already cropped), `none` (raw resize, smoke tests).
Reference: 12 s clip on an M-series CPU → RTF ≈ 0.2, algorithmic latency 1.44 s.

## Patient fine-tuning

```bash
# 1. Re-tokenize labels (old CSVs carry unigram5000 ids; the streaming model uses 1023-piece SPM)
python scripts/regenerate_patient_labels.py --old-csv labels/train.csv labels/val.csv

# 2. Fine-tune from the pretrained bootstrap
python train.py --model-source bootstrap --seed 1 \
  --root-dir /path/to/patient_data \
  --train-file labels/train_spm1023.csv --val-file labels/val_spm1023.csv \
  --precision bf16-mixed --batch-size 2 --accumulate-grad-batches 4

# 3. Evaluate in streaming mode
python eval.py --checkpoint exp/run/last.ckpt --mode streaming \
  --root-dir /path/to/patient_data --test-file labels/test_spm1023.csv --preprocess roi
```

LoRA variant: add `--lora --lora-r 8 --lora-alpha 16 --lora-scopes all` (trains ~2.4% of
parameters). The run directory gets `model_lora_merged.pth`, which `eval.py` and
`demo_realtime.py` consume directly. On the cluster: `sbatch slurm/train_realtime_{finetune,lora}.sbatch`.

**Four things that will bite you:**

- **Always pass `--seed` and compare medians over ≥3 seeds.** Unseeded, this recipe has ~24pp
  run-to-run WER variance; a single run means nothing.
- **Report `corpus_wer`, not `avg_wer`.** `eval.py` emits both; `corpus_wer` is pooled (total edits
  / total reference words, the standard convention) and is what `asr_baselines/metrics.py` uses.
  The two differ by a few points, so mixing them invalidates any cross-model comparison.
- **Select checkpoints in streaming mode.** The default merges whatever the last epoch produced.
  `val_loss` anti-correlates with WER here, and greedy-utterance `val_wer` does not track streaming
  WER either. Train with `AVG_LAST_N=20`, then
  `RUN_DIR=<dir> sbatch slurm/select_best_streaming_epoch.sbatch` to decode every candidate epoch
  in real streaming mode and report on a disjoint set. Worth 17–42pp.
- **Do not re-run face detection on already-cropped clips** — use `--preprocess roi` end to end.
  Fine-tunes from the bootstrap must keep `spm/spm_unigram_1023.model` (enforced; sha256 recorded
  in the run manifest).

RNN-T loss memory grows with sequence × target length; patient CSVs are 24 s-segmented and
`--max-frames 600` drops outliers. Use batch 1–2 with gradient accumulation.

## Cluster (NYU HPC)

The sbatch templates expect a conda env `avsr_realtime` inside a singularity overlay, created once
with the overlay mounted read-write (run templates mount it `:ro`):

```bash
singularity exec --overlay /scratch/$USER/avsr/overlay-15GB-500K.ext3:rw \
  /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash
source /ext3/env.sh
conda create -y -n avsr_realtime python=3.12 && conda activate avsr_realtime
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install "pytorch-lightning>=2.4,<3" sentencepiece "mediapipe<=0.10.21" opencv-python scikit-image av soundfile
```

Patient data is resolved by `slurm/_patient_data.sh` via `RUN_DATA_MODE`
(`legal_only` | `legacy_only` | `merged`), which also re-tokenizes labels to the 1023-piece vocab
before training. Templates take `PROJECT_ROOT`, `ROOT_DIR`, `RUN_DATA_MODE`, `MAX_STEPS`,
`MAX_EPOCHS`, `SEED`, `LORA_*` as env-var overrides.

One-command reproducers for the two best audio results:
`slurm/whisper_splitv1_best.sbatch` and `slurm/nemotron_splitv1_best.sh`.

## How streaming works

Frames are consumed in fixed windows: `lookback` past frames (frontend receptive field, trimmed
from the features), the new `segment`, and `right-context` lookahead. `Emformer.infer` requires
**exactly** segment + right-context fused frames per call; `RNNTBeamSearch.infer` carries encoder
state and the beam hypothesis across calls (`--no-carry-state` reproduces the tutorial's per-chunk
reset). Algorithmic latency = (segment + rc)/25 fps; per-chunk wall-clock RTF is reported by the
demo and by `eval.py`.

## Provenance / licenses

- Recipe code ported from [pytorch/audio examples/avsr](https://github.com/pytorch/audio/tree/main/examples/avsr) (BSD-2-Clause).
- `preparation/detectors/mediapipe_face/` is torchaudio's data_prep mediapipe detector; the
  original auto_avsr detectors (mouth ROIs, Apache 2.0) are kept alongside.
- Pretrained weights: torchaudio `tutorial-assets/device_avsr_model.pt` (BSD-2-Clause), trained on
  LRS3 + VoxCeleb2 + AVSpeech per Ma et al., "Auto-AVSR" (ICASSP 2023).
