#!/bin/bash
# CPU job: build /scratch/pa2753/envs/nemo3 (NeMo Speech 3.x) for nvidia/nemotron-labs-unified-asr-0.6b,
# download the checkpoint into HF_HOME and check that it restores. nemo_asr (NeMo 2.7.3) is not touched.
#SBATCH --job-name=nemo3_env
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
set -euo pipefail
ENV=/scratch/pa2753/envs/nemo3
export UV_CACHE_DIR=/scratch/pa2753/tmp/uv HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1
UV=/home/pa2753/.local/bin/uv
[ -x $ENV/bin/python ] || $UV venv --python 3.12 $ENV
$UV pip install --python $ENV/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu130
$UV pip install --python $ENV/bin/python "nemo_toolkit[asr]>=3.0.0" || \
  $UV pip install --python $ENV/bin/python "nemo_toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git"
$ENV/bin/python - <<'PY'
import importlib.metadata as m, inspect, torch
print("nemo", m.version("nemo_toolkit"), "torch", torch.__version__, torch.version.cuda)
import nemo.collections.asr as nemo_asr
for name in ("nvidia/nemotron-labs-unified-asr-0.6b", "nvidia/nemotron-speech-streaming-en-0.6b"):
    mdl = nemo_asr.models.ASRModel.from_pretrained(name, map_location="cpu")
    print(name, type(mdl).__name__, "target_lang" in inspect.signature(mdl.transcribe).parameters)
PY
echo "done $(date)"
