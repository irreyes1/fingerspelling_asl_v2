#!/usr/bin/env bash
set -euo pipefail

# Reproduce run config: tcn_bilstm_gpu_lightning_20260307
# Source run metadata date: 2026-03-07 (W&B run id: dpor4ykv)

DATA_DIR="${1:-/teamspace/studios/this_studio/fingerspelling_asl_v2/data/kaggle}"

python -m src.train \
  --data_dir "$DATA_DIR" \
  --arch tcn_bilstm \
  --epochs 30 \
  --batch_size 16 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --hidden_dim 256 \
  --num_layers 2 \
  --bidirectional \
  --max_frames 160 \
  --use_delta_features \
  --num_workers 2 \
  --prefetch_factor 2 \
  --early_stopping_patience 8 \
  --early_stopping_min_delta 0.001 \
  --train_size 0 \
  --val_size 0 \
  --use_wandb \
  --wandb_project fingerspelling_asl \
  --wandb_mode online \
  --wandb_run_name tcn_bilstm_gpu_lightning_20260307
