#!/usr/bin/env bash
# Evaluation.

DATA_DIR="${DATA_DIR:-/path/to/CC-CCII_public/}"
CKPT="${1:-./runs/probssl/model.pt}"
RUN_DIR="$(dirname "$CKPT")"
FOLD="${FOLD:-0}"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python eval.py \
  --data_dir "${DATA_DIR}" \
  --csv_list ./csv/ \
  --fold "${FOLD}" \
  --pretrained_dir "${RUN_DIR}" \
  --pretrained_model_name "$(basename "$CKPT")" \
  2>&1 | tee "${RUN_DIR}/val.log"
