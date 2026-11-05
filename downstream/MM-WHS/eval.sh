#!/usr/bin/env bash
# Evaluation.

DATA_DIR="${DATA_DIR:-/path/to/MM-WHS}"
CKPT="${1:-./runs/probssl/model.pt}"
RUN_DIR="$(dirname "$CKPT")"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python val.py \
  --json_list ./dataset.json \
  --data_dir "${DATA_DIR}" \
  --feature_size 48 \
  --infer_overlap 0.75 \
  --pretrained_dir "${RUN_DIR}" \
  --pretrained_model_name "$(basename "$CKPT")" \
  2>&1 | tee "${RUN_DIR}/val.log"
