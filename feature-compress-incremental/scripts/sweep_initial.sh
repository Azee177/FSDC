#!/usr/bin/env bash
set -euo pipefail

# Arguments:
#   1: path config (default: configs/paths.yaml)
#   2: task config (default: configs/coco_10cls.yaml)
#   3: backbone (default: ViT-B-16)
#   4: pretrained weights identifier/path (default: openai)
#   5: epochs (default: 5)
#   6: batch size (default: 128)
#   7: learning rate (default: 1e-3)

PATHS_CFG=${1:-configs/paths.yaml}
TASK_CFG=${2:-configs/coco_10cls.yaml}
BACKBONE=${3:-"ViT-B-16"}
PRETRAINED=${4:-"openai"}
EPOCHS=${5:-5}
BATCH=${6:-128}
LR=${7:-1e-3}

COMPRESS_LIST=${COMPRESS_LIST:-"2 4 8 10 16 32 64 128"}

echo "[sweep_initial] running n_compress sweep: ${COMPRESS_LIST}"

for DIM in ${COMPRESS_LIST}; do
  SAVE_DIR="outputs/compress${DIM}_initial"
  echo "[sweep_initial] -> n_compress=${DIM}, save_dir=${SAVE_DIR}"
  python -u -m src.train_initial \
    --paths "${PATHS_CFG}" \
    --task_cfg "${TASK_CFG}" \
    --backbone "${BACKBONE}" \
    --pretrained "${PRETRAINED}" \
    --n_compress "${DIM}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --save_dir "${SAVE_DIR}" \
    --use_cache
done
