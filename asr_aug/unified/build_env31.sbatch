#!/bin/bash
# CPU job: /scratch/pa2753/envs/nemo31 = NeMo Speech main (3.1.0, the version the unified model card names), editable
# from /scratch/pa2753/unified/Speech (user-approved 2026-10-28). nemo3 (release 3.0.0) is kept for comparison.
#SBATCH --job-name=nemo31_env
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
set -euo pipefail
ENV=/scratch/pa2753/envs/nemo31
export UV_CACHE_DIR=/scratch/pa2753/tmp/uv UV_PYTHON_INSTALL_DIR=/scratch/pa2753/tmp/uv-python HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1
UV=/home/pa2753/.local/bin/uv
cd /scratch/pa2753/unified/Speech && git log -1 --format="NeMo main %h %cd"
# uv-managed CPython: the system /usr python3.12 has no Python.h, which Triton needs to build its CUDA driver stub
# (first nemo31 build failed at decode time with "Python.h: No such file or directory").
$ENV/bin/python -c "import sys; sys.exit(sys.base_prefix == '/usr')" 2>/dev/null || { rm -rf $ENV; $UV venv --managed-python --python 3.12 $ENV; }
$ENV/bin/python -c "import sysconfig, os; p = sysconfig.get_paths()['include']; print('include', p); assert os.path.exists(p + '/Python.h')"
$UV pip install --python $ENV/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu130
$UV pip install --python $ENV/bin/python -e "/scratch/pa2753/unified/Speech[asr]"
HF_HUB_OFFLINE=1 $ENV/bin/python - <<'PY'
import importlib.metadata as m, inspect, torch
import nemo.collections.asr as na
print("nemo", m.version("nemo_toolkit"), "torch", torch.__version__)
M = na.models.ASRModel.from_pretrained("nvidia/nemotron-labs-unified-asr-0.6b", map_location="cpu")
print(type(M).__name__, "target_lang param:", "target_lang" in inspect.signature(M.transcribe).parameters)
print("en prompts:", {k: v for k, v in (M.cfg.model_defaults.get("prompt_dictionary") or {}).items() if k.startswith("en")})
PY
echo "done $(date)"
