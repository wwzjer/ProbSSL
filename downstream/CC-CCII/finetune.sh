#!/usr/bin/env bash
# CC-CCII classification.

DATA_DIR="${DATA_DIR:-/path/to/CC-CCII_public/}"
SSL_CKPT="${SSL_CKPT:-../../pretrain/runs/probssl/model_bestValRMSE.pt}"
LOGDIR="${LOGDIR:-probssl}"
ARCH="${ARCH:-unetr}"
FOLD="${FOLD:-0}"
OPTIM_LR="${OPTIM_LR:-3e-4}"
REG_WEIGHT="${REG_WEIGHT:-1e-5}"
DROPOUT_PATH_RATE="${DROPOUT_PATH_RATE:-0.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_DIR="./runs/${LOGDIR}"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 python -u main.py \
  --data_dir "${DATA_DIR}" \
  --csv_list ./csv/ \
  --fold "${FOLD}" \
  --arch "${ARCH}" \
  --pretrained_checkpoint "${SSL_CKPT}" \
  --use_ssl_pretrained \
  --feature_size 48 \
  --batch_size 4 \
  --max_epochs 100 \
  --optim_lr "${OPTIM_LR}" \
  --reg_weight "${REG_WEIGHT}" \
  --dropout_path_rate "${DROPOUT_PATH_RATE}" \
  --lrschedule warmup_cosine \
  --per_step_lr \
  ${EXTRA_ARGS} \
  --logdir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/finetune.log"
