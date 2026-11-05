#!/usr/bin/env bash
# BraTS Task01 (387/97).
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/Task01_BrainTumour}"
SSL_CKPT="${SSL_CKPT:-../../pretrain/runs/probssl/model_bestValRMSE.pt}"
LOGDIR="${LOGDIR:-probssl}"
CACHE_DIR="${CACHE_DIR:-./cache}"
OPTIM_LR="${OPTIM_LR:-3e-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_DIR="./runs/${LOGDIR}"
mkdir -p "${RUN_DIR}" "${CACHE_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 python -u main.py \
  --json_list ./jsons/task01_brats.json \
  --data_dir "${DATA_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --feature_size 48 \
  --in_channels 4 \
  --out_channels 3 \
  --use_ssl_pretrained \
  --ssl_pretrained_ckpt "${SSL_CKPT}" \
  --roi_x 128 --roi_y 128 --roi_z 128 \
  --use_checkpoint \
  --batch_size 1 \
  --sw_batch_size 2 \
  --max_epochs 300 \
  --val_every 5 \
  --optim_lr "${OPTIM_LR}" \
  --warmup_epochs 5 \
  --lrschedule warmup_cosine \
  --per_step_lr \
  --infer_overlap 0.75 \
  --save_checkpoint \
  ${EXTRA_ARGS} \
  --logdir "${LOGDIR}" 2>&1 | tee "${RUN_DIR}/finetune.log"
