#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir="${1:-outputs/smoke}"
python -m unittest discover -s tests -v
python scripts/validate_data.py
python -m tx2mol.pretrain --config configs/pretrain.json \
  --data_path data/examples/train.csv --epochs 2 --max_steps 2 \
  --output_dir "$run_dir/gene_vae" --device cpu
python -m tx2mol.finetune --config configs/finetune.json \
  --data_path data/examples/train.csv --val_data_path data/examples/val.csv \
  --saved_gene_vae "$run_dir/gene_vae/gene_vae.pt" \
  --output_dir "$run_dir/tx2mol" --epochs 1 --batch_size 2 \
  --max_train_steps 2 --max_eval_batches 1 --skip_generation_eval
python -m tx2mol.generate --config configs/generate.json \
  --model_dir "$run_dir/tx2mol" --saved_gene_vae "$run_dir/gene_vae/gene_vae.pt" \
  --output_dir "$run_dir/generated" --num_runs 1 --num_samples 2 --batch_size 2
printf '\nThree-stage smoke test completed: %s\n' "$run_dir"
