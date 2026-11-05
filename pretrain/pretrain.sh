#!/usr/bin/env bash
# ProbSSL pretrain.

DATA_PATH="${DATA_PATH:-/path/to/pretrain_data/}"
JSON_PATH="${JSON_PATH:-./jsons/}"
GPUS="${GPUS:-0,1}"
LOGDIR="${LOGDIR:-probssl}"
RUN_DIR="./runs/${LOGDIR}"
mkdir -p "${RUN_DIR}"

export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=${GPUS} PYTHONUNBUFFERED=1 \
python -m torch.distributed.run --nproc_per_node=2 --master_port=11223 main.py \
  --data_path "${DATA_PATH}" --json_path "${JSON_PATH}" \
  --batch_size=1 --num_steps=250000 --eval_num=500 --warmup_steps=5000 \
  --lrdecay --lr=6e-6 --decay=0.1 --opt adamw --lr_schedule warmup_cosine \
  --logdir "${LOGDIR}" \
  2>&1 | tee "${RUN_DIR}/pretrain.log"
