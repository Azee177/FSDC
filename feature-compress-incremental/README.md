# Quick Start
1. `pip install -r requirements.txt`
2. `cp configs/paths_example.yaml configs/paths.yaml` and edit it to match your actual paths (e.g. `/big-data/dataset-academic/COCO/train2017`).
3. `bash scripts/prepare_coco_cls.sh configs/paths.yaml configs/coco_10cls.yaml`

## Repository Layout
- `brief.md`: original concept note (unchanged)
- `configs/`: YAML configs for tasks, paths, and incremental settings
- `src/`: modular code for data, models, training, baselines, and visualization
- `scripts/`: helper shell scripts for preparation and experiments
- `outputs/`: default directory for logs, checkpoints, and cached features

## How to Run
### Build COCO classification splits
```bash
bash scripts/prepare_coco_cls.sh configs/paths.yaml configs/coco_10cls.yaml
# or python scripts/prepare_coco_cls.py --paths configs/paths.yaml --task configs/coco_10cls.yaml
```
- Produces `train_list.txt`, `val_list.txt`, and `class_map.json`
- When `configs/coco_add_cls.yaml` defines `new_class_ids`, it also emits `new_train_list.txt` and `new_val_list.txt`

### Baseline: linear probe
```bash
python -u src/baselines/linear_probe.py   --paths configs/paths.yaml   --task_cfg configs/coco_10cls.yaml   --backbone "ViT-B-16"   --epochs 5 --batch_size 128 --lr 1e-3   --save_dir outputs/linear_probe --use_cache
```

### Proposed method: compression head + classifier
```bash
python -u src/train_initial.py   --paths configs/paths.yaml   --task_cfg configs/coco_10cls.yaml   --backbone "ViT-B-16"   --n_compress 32   --epochs 5 --batch_size 128 --lr 1e-3   --save_dir outputs/compress32_initial --use_cache
```

### Incremental add-one-class
```bash
python -u src/add_class_incremental.py   --paths configs/paths.yaml   --task_cfg configs/coco_add_cls.yaml   --backbone "ViT-B-16"   --resume_ckpt outputs/compress32_initial/best.ckpt   --n_compress 32   --epochs 1 --batch_size 128 --lr 1e-3   --save_dir outputs/compress32_addcls --use_cache
```

### Incremental baseline: finetune + replay
```bash
python -u src/baselines/finetune_replay.py   --paths configs/paths.yaml   --task_cfg configs/coco_add_cls.yaml   --backbone "ViT-B-16"   --resume_ckpt outputs/linear_probe/best.ckpt   --epochs 1 --batch_size 128 --lr 1e-3   --save_dir outputs/finetune_replay --use_cache
```

### One-click scripts
- `bash scripts/run_all_initial.sh`: run linear probe then compression training in sequence
- `bash scripts/run_add_class.sh`: run incremental method and finetune baseline

## Outputs
- `metrics.json` / `metrics_incremental.json`: top-1/top-5 accuracy, losses, and old/new class metrics
- `confusion_*.png` + `confusion_*.json`: confusion matrices (plot + numbers)
- `summary.json`: parameter counts and best scores
- `outputs/cache/`: cached CLIP features (`.pt`) grouped by split/backbone

## Visualization
```bash
python -u src/viz/inspect_features.py   --paths configs/paths.yaml   --task_cfg configs/coco_10cls.yaml   --backbone "ViT-B-16"   --checkpoint outputs/compress32_initial/best.ckpt   --save_dir outputs/viz
```
Produces `channel_class_heatmap.png` and `per_class_top_channels.json` for analyzing channel responses.

## Dependencies
```bash
pip install -r requirements.txt
```
Requires Python >=3.10 and a CUDA-capable PyTorch build (server has 8x4090, the supplied requirements file works out of the box).

## Configuration Notes
- `configs/paths_example.yaml` is a template; copy it to `configs/paths.yaml` and adjust to your data/model paths (e.g. `/big-data/dataset-academic/COCO/train2017`, `/big-data/public/models/huggingface/hub/models--openai--clip-vit-large-patch14-336`).
- `configs/coco_10cls.yaml` lists the initial ten classes; `configs/coco_add_cls.yaml` defines the incremental class.
- Command-line flags such as `--n_compress` or `--use_patch` can override the YAML defaults (using patches triggers a 1x1 conv compress head).

## Feature Cache & Performance
- `train_initial.py`, `linear_probe.py`, `add_class_incremental.py`, and `finetune_replay.py` write CLIP features to `outputs/cache/` to avoid repeated forward passes.
- Re-run with `--use_cache` to reuse the `.pt` feature dumps instead of re-encoding images.

## Remarks
- `brief.md` originally described a dataclass-based config flow; this repository now follows the YAML + shell workflow required by the latest blueprint.
- All Python files contain per-line Chinese comments (as required) and avoid recursion; deploy under `/big-data/person/xiaozeyu/FSDC` on the server.
