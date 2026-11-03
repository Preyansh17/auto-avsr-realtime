#!/bin/bash
# A0 smoke test (asr_aug/gen/README.md): do both IndexTTS envs run on a B200?
#   idx2: IndexTTS-2 and stock IndexTTS-1.5 each speak one sentence in semg1's and semg2's voice (ft_stitch5 refs).
#   pdis: ProtoDisent's patched UnifiedVoice loads IndexTTS-1.5 gpt.pth and runs a forward pass on the GPU.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 smoke.sbatch
#SBATCH --job-name=gen_smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -uo pipefail
G=/scratch/pa2753/gen; O=$G/smoke; mkdir -p $O
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"

cd $G/code/index-tts
/scratch/pa2753/envs/idx2/bin/python - <<EOF
import time, torch
print("torch", torch.__version__, "arch", torch.cuda.get_arch_list(), "cap", torch.cuda.get_device_capability())
TEXT = "Please turn the light on when you come back from the kitchen."
REFS = {p: f"/scratch/pa2753/stack/higgs/{p}/refs/ft_stitch5.wav" for p in ("s1", "s2")}
from indextts.infer_v2 import IndexTTS2
t = time.time()
m = IndexTTS2(cfg_path="$G/models/IndexTTS-2/config.yaml", model_dir="$G/models/IndexTTS-2", use_fp16=True, use_cuda_kernel=False)
print(f"IndexTTS-2 loaded {time.time() - t:.0f}s")
for p, ref in REFS.items():
    t = time.time(); m.infer(spk_audio_prompt=ref, text=TEXT, output_path=f"$O/idx2_{p}.wav"); print(f"idx2 {p} {time.time() - t:.1f}s")
del m; torch.cuda.empty_cache()
from indextts.infer import IndexTTS
m = IndexTTS(cfg_path="$G/models/IndexTTS-1.5/config.yaml", model_dir="$G/models/IndexTTS-1.5", use_fp16=True, use_cuda_kernel=False)
for p, ref in REFS.items():
    t = time.time(); m.infer(ref, TEXT, f"$O/idx15_{p}.wav"); print(f"idx15 {p} {time.time() - t:.1f}s")
print("max mem GB", torch.cuda.max_memory_allocated() / 1e9)
EOF
echo "idx2 exit $?"

cd $G/code/Controllable-Dysarthric-Speech-Synthesis
/scratch/pa2753/envs/pdis/bin/python - <<EOF
import torch
from omegaconf import OmegaConf
print("torch", torch.__version__, "arch", torch.cuda.get_arch_list())
from train import load_UnifiedVoice
cfg = OmegaConf.load("configs/controllable_dysarthric_speech_synthesis.yaml")
m = load_UnifiedVoice(cfg.gpt, "$G/models/IndexTTS-1.5/gpt.pth", "cuda")
x = torch.randn(4096, 4096, device="cuda"); print("matmul ok", float((x @ x).mean()))
print("UnifiedVoice params M", sum(p.numel() for p in m.parameters()) / 1e6)
EOF
echo "pdis exit $?"
ls -la $O; echo "done $(date)"
