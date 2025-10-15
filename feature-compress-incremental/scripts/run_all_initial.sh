#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH=${1:-configs/paths.yaml}
TASK_CFG=${2:-configs/coco_10cls.yaml}
BACKBONE=${3:-ViT-B-16}
python -u src/baselines/linear_probe.py --paths "$CONFIG_PATH" --task_cfg "$TASK_CFG" --backbone "$BACKBONE" --save_dir outputs/linear_probe --use_cache
python -u src/train_initial.py --paths "$CONFIG_PATH" --task_cfg "$TASK_CFG" --backbone "$BACKBONE" --save_dir outputs/compress_initial --use_cache
