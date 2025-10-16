#!/usr/bin/env bash
set -euo pipefail

# Arguments:
#   1: path config (default: configs/paths.yaml)
#   2: backbone (default: ViT-B-16)
#   3: pretrained weights identifier/path (default: openai)
#   4: initial checkpoint (default: outputs/compress32_initial/best.ckpt)
#   5: base task config (default: configs/coco_10cls.yaml) -- used only for reference
#
# The script expects the following incremental configs to exist:
#   - configs/coco_add_cls.yaml        (stage 1, +1 class)
#   - configs/coco_add_cls_stage2.yaml (stage 2, +2 classes)
#   - configs/coco_add_cls_stage3.yaml (stage 3, +3 classes)

PATHS_CFG=${1:-configs/paths.yaml}
BACKBONE=${2:-"ViT-B-16"}
PRETRAINED=${3:-"openai"}
INITIAL_CKPT=${4:-outputs/compress32_initial/best.ckpt}

STAGE1_CFG=${STAGE1_CFG:-configs/coco_add_cls.yaml}
STAGE2_CFG=${STAGE2_CFG:-configs/coco_add_cls_stage2.yaml}
STAGE3_CFG=${STAGE3_CFG:-configs/coco_add_cls_stage3.yaml}

STAGE1_DIR=${STAGE1_DIR:-outputs/compress32_addcls_stage1}
STAGE2_DIR=${STAGE2_DIR:-outputs/compress32_addcls_stage2}
STAGE3_DIR=${STAGE3_DIR:-outputs/compress32_addcls_stage3}

echo "[run_incremental_stages] stage 1 -> ${STAGE1_CFG}"
python -u -m src.add_class_incremental \
  --paths "${PATHS_CFG}" \
  --task_cfg "${STAGE1_CFG}" \
  --backbone "${BACKBONE}" \
  --pretrained "${PRETRAINED}" \
  --resume_ckpt "${INITIAL_CKPT}" \
  --n_compress 32 \
  --epochs 1 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_dir "${STAGE1_DIR}" \
  --use_cache

STAGE1_BEST="${STAGE1_DIR}/best_incremental.ckpt"

echo "[run_incremental_stages] stage 2 -> ${STAGE2_CFG}"
python -u -m src.add_class_incremental \
  --paths "${PATHS_CFG}" \
  --task_cfg "${STAGE2_CFG}" \
  --backbone "${BACKBONE}" \
  --pretrained "${PRETRAINED}" \
  --resume_ckpt "${STAGE1_BEST}" \
  --n_compress 32 \
  --epochs 1 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_dir "${STAGE2_DIR}" \
  --use_cache

STAGE2_BEST="${STAGE2_DIR}/best_incremental.ckpt"

echo "[run_incremental_stages] stage 3 -> ${STAGE3_CFG}"
python -u -m src.add_class_incremental \
  --paths "${PATHS_CFG}" \
  --task_cfg "${STAGE3_CFG}" \
  --backbone "${BACKBONE}" \
  --pretrained "${PRETRAINED}" \
  --resume_ckpt "${STAGE2_BEST}" \
  --n_compress 32 \
  --epochs 1 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_dir "${STAGE3_DIR}" \
  --use_cache


