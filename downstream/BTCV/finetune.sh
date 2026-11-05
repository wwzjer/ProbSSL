#!/usr/bin/env bash
# Downstream fine-tuning.

DATA_DIR="${DATA_DIR:-/path/to/BTCV}"
SSL_CKPT="${SSL_CKPT:-../../pretrain/runs/probssl/model_bestValRMSE.pt}"
LOGDIR="${LOGDIR:-probssl}"
RUN_DIR="./runs/${LOGDIR}"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
  --json_list dataset_0.json \
  --data_dir "${DATA_DIR}" \
  --feature_size 48 \
  --use_ssl_pretrained \
  --ssl_pretrained_ckpt "${SSL_CKPT}" \
  --roi_x 96 --roi_y 96 --roi_z 96 \
  --use_checkpoint \
  --batch_size 1 \
  --max_epochs 5000 \
  --optim_lr 1e-4 \
  --reg_weight 0.05 \
  --dropout_path_rate 0.2 \
  --lrschedule warmup_cosine \
  --infer_overlap 0.75 \
  --save_checkpoint \
  --logdir "${LOGDIR}" 2>&1 | tee "${RUN_DIR}/finetune.log"
