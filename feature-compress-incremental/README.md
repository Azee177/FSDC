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

### Incremental baseline: LwF
```bash
python -u src/baselines/lwf.py   --paths configs/paths.yaml   --task_cfg configs/coco_add_cls.yaml   --backbone "ViT-B-16"   --resume_ckpt outputs/linear_probe/best.ckpt   --epochs 1 --batch_size 128 --lr 1e-3   --temperature 2.0 --alpha 0.7   --save_dir outputs/lwf --use_cache
```

### Incremental baseline: iCaRL
```bash
python -u src/baselines/icarl.py   --paths configs/paths.yaml   --task_cfg configs/coco_add_cls.yaml   --backbone "ViT-B-16"   --resume_ckpt outputs/linear_probe/best.ckpt   --epochs 1 --batch_size 128 --lr 1e-3   --memory_per_class 20 --save_dir outputs/icarl --use_cache
```

### Incremental baseline: FOSTER
```bash
python -u src/baselines/foster.py   --paths configs/paths.yaml   --task_cfg configs/coco_add_cls.yaml   --backbone "ViT-B-16"   --resume_ckpt outputs/linear_probe/best.ckpt   --epochs 1 --batch_size 128 --lr 1e-3   --replay_per_class 10 --beta 0.5 --save_dir outputs/foster --use_cache
```

### Visualization & Analysis
```bash
# Channel-class response heatmap (existing)
python -u src/viz/inspect_features.py   --paths configs/paths.yaml   --task_cfg configs/coco_10cls.yaml   --backbone "ViT-B-16"   --checkpoint outputs/compress32_initial/best.ckpt   --save_dir outputs/viz

# t-SNE / UMAP on cached features
python -u src/viz/tsne_umap.py   --paths configs/paths.yaml   --task_cfg configs/coco_add_cls.yaml   --feature_cache outputs/cache/ViT-B-16_size224_train.pt   --use_tsne --use_umap   --save_dir outputs/viz

# Parameter & storage analysis
python -u src/utils/params_analysis.py   --paths configs/paths.yaml   --task_cfg configs/coco_10cls.yaml   --backbone "ViT-B-16"   --pretrained openai   --n_compress 32   --class_count 10   --save_path outputs/analysis/params.json
```

### One-click / sweep scripts
- `bash scripts/run_all_initial.sh`: run linear probe then compression training in sequence
- `bash scripts/run_add_class.sh`: run incremental method and finetune baseline
- `bash scripts/sweep_initial.sh [paths_cfg] [task_cfg] ...`: scan `n_compress` values (default `8 16 32 64 128`) for `train_initial.py`
- `bash scripts/sweep_replay.sh [paths_cfg] [task_cfg] ...`: scan `replay_per_class` values (default `0 5 10 20`) for both `add_class_incremental.py` and `finetune_replay.py`
- `bash scripts/run_incremental_stages.sh [paths_cfg] ...`: chain three incremental stages (`coco_add_cls*.yaml`) while reusing the best checkpoint from the previous stage

## Outputs
- metrics.json / metrics_incremental.json / metrics_finetune.json: aggregate accuracy (overall/old/new), delta-old/delta-new (if baseline supplied), forget rate, best top-1, replay/distillation hyper-parameters
- `per_class_accuracy.json`: per-class hit rate derived from the final confusion matrix
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
- `configs/coco_10cls.yaml` lists the initial ten classes; incremental stages are described by `configs/coco_add_cls.yaml`, `configs/coco_add_cls_stage2.yaml`, and `configs/coco_add_cls_stage3.yaml`.
- Command-line flags such as `--n_compress` or `--use_patch` can override the YAML defaults (using patches triggers a 1x1 conv compress head).

## Feature Cache & Performance
- `train_initial.py`, `linear_probe.py`, `add_class_incremental.py`, and `finetune_replay.py` write CLIP features to `outputs/cache/` to avoid repeated forward passes.
- Re-run with `--use_cache` to reuse the `.pt` feature dumps instead of re-encoding images.

## Remarks
- `brief.md` originally described a dataclass-based config flow; this repository now follows the YAML + shell workflow required by the latest blueprint.
- All Python files contain per-line Chinese comments (as required) and avoid recursion; deploy under `/big-data/person/xiaozeyu/FSDC` on the server.

## Experiment Matrix

| Stage | Script / Config | Key Variants | Primary Outputs |
| ----- | ---------------- | ------------ | --------------- |
| Data prep | `scripts/prepare_coco_cls.sh` + `configs/coco_10cls.yaml` | Fixed 10-class subset | `indices/*.txt`, `indices/class_map.json` |
| Initial training | `src/train_initial.py` (`n_compress=8/16/32/64/128`) or `scripts/sweep_initial.sh` | Compression width / optional `--use_patch` | `outputs/compress{k}_initial/metrics.json`, `best.ckpt` |
- metrics.json / metrics_incremental.json / metrics_finetune.json: aggregate accuracy (overall/old/new), delta-old/delta-new (if baseline supplied), forget rate, best top-1, replay/distillation hyper-parameters
| Baselines | `src/baselines/finetune_replay.py`, `src/baselines/lwf.py`, `src/baselines/icarl.py`, `src/baselines/foster.py`, `scripts/sweep_replay.sh` | Distillation temps/weights, exemplar capacity | `outputs/{baseline}/metrics_*.json`, `confusion_*.png/json` |
| Visualisation & analysis | `src/viz/inspect_features.py`, `src/viz/tsne_umap.py`, `src/utils/params_analysis.py` | t-SNE / UMAP, parameter & storage analysis | `outputs/viz/*.png`, `outputs/analysis/*.json` |

## Metric Definitions

- `best_top1`: best validation Top-1 during training.
- `overall_acc`: `trace(confusion) / sum(confusion)` from the final confusion matrix.
- `old_acc` / `new_acc`: recall on the base / incremental class sets.
- `delta_old` / `delta_new`: accuracy deltas versus a chosen baseline (e.g. initial model or linear probe).
- `forget_rate`: `max(0, (baseline_old_acc - old_acc) / baseline_old_acc)`.
- `per_class_accuracy.json`: per-class recall values for further plotting.
- `temperature`, `alpha`, `beta`, `gamma`, `replay_per_class`, `memory_per_class`: distillation / replay hyper-parameters captured in baseline metrics.

See `reports/improve_progress.md` for consolidated tables and discussion.

## Resource & Runtime Notes

- Hardware: 8× RTX 4090 (all experiments run on a single GPU unless noted).
- Feature extraction timings recorded by the built-in timer (from `log.txt`):

| Operation | Experiment | Duration (s) | Notes |
| --------- | ---------- | ------------ | ----- |
| Extract `train` features | `linear_probe` / `train_initial` | 1251.30 | ~20.9 min; cached for later stages |
| Extract `val` features | same as above | 211.76 | ~3.5 min |
| Extract `train_new` features | `coco_add_cls.yaml` incremental run | 6.72 | only the new class |
| Extract `val_new` features | same incremental run | 1.94 | validation subset for the new class |

- Keep the `loguru` outputs when running sweep scripts to capture training iteration durations under different replay sizes.
