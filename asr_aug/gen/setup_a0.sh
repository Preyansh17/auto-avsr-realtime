#!/bin/bash
# A0 setup for the IndexTTS generator benchmark (asr_aug/gen/README.md). Run on the login node (needs internet):
#   nohup bash setup_a0.sh > /scratch/pa2753/gen/logs/setup_a0.log 2>&1 &
# Envs: idx2 = index-tts (IndexTTS-2) + peft for G1 (risingore LoRA); pdis = ProtoDisent (IndexTTS-1.5 + Seed-VC) for G2.
# Each part is skipped when its done-marker exists, so a rerun picks up where a failed one stopped.
set -uo pipefail
G=/scratch/pa2753/gen; E=/scratch/pa2753/envs; C=$G/code
export UV_CACHE_DIR=/scratch/pa2753/uv_cache HF_HOME=/scratch/pa2753/hf_cache UV_PYTHON_INSTALL_DIR=/scratch/pa2753/uv_python
step() { [ -f $G/logs/done_$1 ] && { echo "skip $1"; return; }; echo "== $1 $(date)"; if "${@:2}"; then touch $G/logs/done_$1; echo "ok $1"; else echo "FAIL $1"; fi; }

env_idx2() { cd $C/index-tts && UV_PROJECT_ENVIRONMENT=$E/idx2 uv sync --python 3.11 --no-dev && \
  uv pip install --python $E/idx2/bin/python "peft==0.17.*"; }
# webrtcvad (via resemblyzer) needs a C compiler, which the login node lacks: swap in the prebuilt webrtcvad-wheels
# (same module). The repo pins transformers==4.46.3 (its vendored generation utils are copied from it) and tokenizers==0.21.0, which
# 4.46.3 does not accept (<0.21); keep transformers and override tokenizers to the newest version it accepts
env_pdis() { cd $C/Controllable-Dysarthric-Speech-Synthesis && uv venv --clear --python 3.10 $E/pdis && \
  printf 'tokenizers==0.20.3\nwebrtcvad ; sys_platform == "never"\n' > $G/logs/pdis_override.txt && \
  uv pip install --python $E/pdis/bin/python --torch-backend cu128 --override $G/logs/pdis_override.txt \
    -e . -r requirements-seed-vc.txt webrtcvad-wheels; }
w_idx2() { $E/idx2/bin/hf download IndexTeam/IndexTTS-2 --local-dir $G/models/IndexTTS-2; }
w_idx15() { $E/idx2/bin/hf download IndexTeam/IndexTTS-1.5 --local-dir $G/models/IndexTTS-1.5; }
# no bzip2 binary on the login node, so tar xj fails: extract with Python's tarfile
torgo_ctl() { mkdir -p $G/data/torgo_dl && cd $G/data/torgo_dl && for f in FC MC; do
    [ -f $f.ok ] || { wget -q -c http://www.cs.toronto.edu/~complingweb/data/TORGO/$f.tar.bz2 && \
      python3 -c "import tarfile,sys; tarfile.open(sys.argv[1], 'r:bz2').extractall()" $f.tar.bz2 && rm $f.tar.bz2 && touch $f.ok; } || return 1; done; }
# TORGO root for prepare_torgo.py: dysarthric speakers symlinked from th3482 (read-only, nothing written there) + controls
torgo_root() { mkdir -p $G/data/TORGO && cd $G/data/TORGO && for s in /scratch/th3482/TORGO/raw/*; do ln -sfn $s .; done && \
  for s in $G/data/torgo_dl/*/*/ $G/data/torgo_dl/*/; do b=$(basename $s); [[ $b =~ ^[FM]C0[0-9]$ ]] && ln -sfn ${s%/} .; done; ls; [ -e FC01 ] && [ -e MC01 ]; }

step env_idx2 env_idx2
step env_pdis env_pdis
step w_idx2 w_idx2
step w_idx15 w_idx15
step torgo_ctl torgo_ctl
step torgo_root torgo_root
# IndexTTS-2 fetches w2v-bert, campplus, MaskGCT and BigVGAN on first start; get them here (compute nodes run offline)
aux_idx2() { cd $C/index-tts && $E/idx2/bin/python -c "from indextts.utils.model_download import ensure_models_available as f; print(f('$G/models/IndexTTS-2'))"; }
step aux_idx2 aux_idx2
echo "all done $(date)"
