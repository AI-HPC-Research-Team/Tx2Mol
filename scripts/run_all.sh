#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/validate_data.py
python -m tx2mol.pretrain --config configs/pretrain.json
python -m tx2mol.finetune --config configs/finetune.json
python -m tx2mol.generate --config configs/generate.json
python scripts/evaluate_release_attempts.py \
  --attempts outputs/targets/raw_attempts.csv \
  --output-dir outputs/targets_evaluation
