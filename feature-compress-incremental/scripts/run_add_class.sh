#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH=${1:-configs/paths.yaml}
TASK_CFG=${2:-configs/coco_add_cls.yaml}
BACKBONE=${3:-ViT-B-16}
python -u src/add_class_incremental.py --paths "$CONFIG_PATH" --task_cfg "$TASK_CFG" --backbone "$BACKBONE" --resume_ckpt outputs/compress_initial/best.ckpt --save_dir outputs/compress_incremental --use_cache
python -u src/baselines/finetune_replay.py --paths "$CONFIG_PATH" --task_cfg "$TASK_CFG" --backbone "$BACKBONE" --resume_ckpt outputs/linear_probe/best.ckpt --save_dir outputs/finetune_replay --use_cache
