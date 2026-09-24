#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
# CUDA toolkit/nvcc is required when no compatible FlashAttention wheel exists.
MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn==2.6.1 --no-build-isolation
python -m pip install --no-deps -e .
python scripts/check_environment.py
