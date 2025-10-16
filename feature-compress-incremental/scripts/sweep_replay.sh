#!/usr/bin/env bash
set -euo pipefail

# Arguments:
#   1: path config (default: configs/paths.yaml)
#   2: task config for add_class (default: configs/coco_add_cls.yaml)
#   3: backbone (default: ViT-B-16)
#   4: pretrained weights identifier/path (default: openai)
#   5: resume checkpoint for add_class (default: outputs/compress32_initial/best.ckpt)
#   6: resume checkpoint for finetune baseline (default: outputs/linear_probe/best.ckpt)

PATHS_CFG=${1:-configs/paths.yaml}
TASK_CFG=${2:-configs/coco_add_cls.yaml}
BACKBONE=${3:-"ViT-B-16"}
PRETRAINED=${4:-"openai"}
RESUME_ADD=${5:-outputs/compress32_initial/best.ckpt}
RESUME_BASELINE=${6:-outputs/linear_probe/best.ckpt}

REPLAY_VALUES=${REPLAY_VALUES:-"0 5 10 20"}

echo "[sweep_replay] running replay_per_class sweep: ${REPLAY_VALUES}"

for REPLAY in ${REPLAY_VALUES}; do
  ADD_SAVE_DIR="outputs/compress32_addcls_replay${REPLAY}"
  BASE_SAVE_DIR="outputs/finetune_replay_${REPLAY}"

  echo "[sweep_replay] -> replay_per_class=${REPLAY}"
  python -u -m src.add_class_incremental \
    --paths "${PATHS_CFG}" \
    --task_cfg "${TASK_CFG}" \
    --backbone "${BACKBONE}" \
    --pretrained "${PRETRAINED}" \
    --resume_ckpt "${RESUME_ADD}" \
    --n_compress 32 \
    --replay_per_class "${REPLAY}" \
    --epochs 1 \
    --batch_size 128 \
    --lr 1e-3 \
    --save_dir "${ADD_SAVE_DIR}" \
    --use_cache

  python -u -m src.baselines.finetune_replay \
    --paths "${PATHS_CFG}" \
    --task_cfg "${TASK_CFG}" \
    --backbone "${BACKBONE}" \
    --pretrained "${PRETRAINED}" \
    --resume_ckpt "${RESUME_BASELINE}" \
    --replay_per_class "${REPLAY}" \
    --epochs 1 \
    --batch_size 128 \
    --lr 1e-3 \
    --save_dir "${BASE_SAVE_DIR}" \
    --use_cache
done
