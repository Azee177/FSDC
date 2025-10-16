# Improve Progress Report

## Overview

This document tracks the staged improvements proposed in `improve.md` and their current implementation status inside `feature-compress-incremental/`.

| Stage | Status | Notes |
| ----- | ------ | ----- |
| Stage 1 – Extended metrics & logging | ✅ Completed (see `src/utils/metrics.py`, new JSON outputs) |
| Stage 2 – Experimental sweeps / multi-stage configs | ✅ Completed (`configs/coco_add_cls_stage*.yaml`, `scripts/sweep_*.sh`, `run_incremental_stages.sh`) |
| Stage 3 – Additional baselines | ✅ Implemented (`src/baselines/lwf.py`, `icarl.py`, `foster.py`) — runs pending |
| Stage 4 – Visualisation & analysis tools | ✅ Implemented (`src/viz/tsne_umap.py`, `src/utils/params_analysis.py`) |
| Stage 5 – Documentation & reporting | ✅ This report + expanded README |

## Key Results (current runs)

| Experiment | Overall Top-1 | Old-class Acc | New-class Acc | Replay / Memory | Notes |
| ---------- | ------------- | ------------- | ------------- | ---------------- | ----- |
| `outputs/compress32_initial` (`train_initial.py`, n=32) | 0.9617 (`metrics.json:best_top1`) | – | – | – | No replay; base model |
| `outputs/compress32_addcls` (`add_class_incremental.py`, replay=5) | 0.7898 | 0.7889 | 0.8099 | 5 per class | Balanced old/new accuracy after incremental step |
| `outputs/finetune_replay` (`finetune_replay.py`, replay=5) | 0.8658 | ≈0.8845 (computed from `confusion_finetune.json`) | ≈0.4373 | 5 per class | Strong old-class retention, new class underfits |

> Old/new accuracies for the finetune baseline were derived from the confusion matrix counts:
> `old_correct = 10665`, `new_correct = 230`, `old_total = 12058`, `new_total = 526`.

Upcoming runs should populate the new baseline directories:

- `outputs/lwf`
- `outputs/icarl`
- `outputs/foster`

Once executed, capture their `metrics_*.json` / `per_class_accuracy.json` and extend the table above.

## Artefacts & Visuals

- Confusion matrices: `outputs/compress32_addcls/confusion_incremental.png`, `outputs/finetune_replay/confusion_finetune.png`.
- Per-class recall JSON: same directories, `per_class_accuracy.json`.
- Planned dimensionality reduction: run `src/viz/tsne_umap.py` to produce `outputs/viz/tsne.png` / `umap.png` per cached feature set.
- Parameter / storage analysis: `src/utils/params_analysis.py --save_path outputs/analysis/params.json` (to be executed after final model selection).

## Resource Notes

- Feature extraction dominates runtime (≈20.9 minutes for the base train set, ≈3.5 minutes for val).
- Incremental feature extraction is negligible (<10 seconds per split).
- Sweeps reuse `outputs/cache/` and should be run with logging enabled to capture iteration-level timings.

## Next Actions

1. **Run new baselines** (LwF, iCaRL, FOSTER) with the same configs as the main method; log metrics and add them to the summary table.
2. **Capture visualisations** (t-SNE / UMAP) for at least one base checkpoint and an incremental stage for qualitative analysis.
3. **Generate parameter/storage report** via `params_analysis.py` once exemplar buffers are available.
4. **Update this report** with the new numbers, figures, and a consolidated comparison ready for manuscript or slide decks.
