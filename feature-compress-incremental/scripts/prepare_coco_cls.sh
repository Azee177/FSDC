#!/usr/bin/env bash
set -e
set -u
set -o pipefail
CONFIG_PATH=$1
TASK_PATH=$2
python - "$CONFIG_PATH" "$TASK_PATH" <<'PY'
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from pycocotools.coco import COCO

paths = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8'))
task = yaml.safe_load(Path(sys.argv[2]).read_text(encoding='utf-8'))

train_list_path = Path(paths['train_list']).expanduser()
val_list_path = Path(paths['val_list']).expanduser()
new_train_path = Path(paths.get('new_train_list', train_list_path.parent / 'new_train_list.txt')).expanduser()
new_val_path = Path(paths.get('new_val_list', val_list_path.parent / 'new_val_list.txt')).expanduser()
class_map_path = Path(paths['class_map']).expanduser()
train_list_path.parent.mkdir(parents=True, exist_ok=True)
val_list_path.parent.mkdir(parents=True, exist_ok=True)
cache_root = Path(paths.get('cache_root', train_list_path.parent)).expanduser()
cache_root.mkdir(parents=True, exist_ok=True)

existing_map = {}
if class_map_path.exists():
    try:
        existing_map = json.loads(class_map_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        existing_map = {}

image_root = Path(paths['coco_root']).expanduser()
annotation_file = Path(paths['annotation_file']).expanduser()
coco = COCO(str(annotation_file))

base_class_ids = [int(cid) for cid in task.get('class_ids', [])]
new_class_ids = [int(cid) for cid in task.get('new_class_ids', [])]
random_seed = task.get('split_seed', 42)
train_ratio = task.get('train_ratio', 0.85)
random.seed(random_seed)

class_map = {str(k): v for k, v in existing_map.items()}

def assign_label_map(ids, starting_map):
    label_map = starting_map.copy()
    next_index = len(label_map)
    for cid in ids:
        key = str(cid)
        if key not in label_map:
            label_map[key] = next_index
            next_index += 1
    return label_map

def build_records(candidate_ids, label_map):
    if not candidate_ids:
        return []
    records = []
    cat_ids = list({int(cid) for cid in candidate_ids})
    for img_id in coco.getImgIds(catIds=cat_ids):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=cat_ids))
        if not anns:
            continue
        areas = defaultdict(float)
        for ann in anns:
            cid = ann['category_id']
            areas[cid] += float(ann.get('area', 0.0))
        if not areas:
            continue
        main_cid, _ = max(areas.items(), key=lambda kv: kv[1])
        key = str(main_cid)
        if key not in label_map:
            continue
        img_path = image_root / coco.imgs[img_id]['file_name']
        records.append((str(img_path), label_map[key]))
    return records

if base_class_ids:
    class_map = assign_label_map(base_class_ids, {})
    records = build_records(base_class_ids, class_map)
    if records:
        random.shuffle(records)
        split = int(len(records) * train_ratio)
        train_records = records[:split]
        val_records = records[split:]
        train_list_path.write_text('
'.join(f"{p} {l}" for p, l in train_records), encoding='utf-8')
        val_list_path.write_text('
'.join(f"{p} {l}" for p, l in val_records), encoding='utf-8')

if new_class_ids:
    class_map = assign_label_map(new_class_ids, class_map)
    records_new = build_records(new_class_ids, class_map)
    if records_new:
        random.shuffle(records_new)
        split_new = int(len(records_new) * train_ratio)
        new_train = records_new[:split_new]
        new_val = records_new[split_new:]
        new_train_path.write_text('
'.join(f"{p} {l}" for p, l in new_train), encoding='utf-8')
        new_val_path.write_text('
'.join(f"{p} {l}" for p, l in new_val), encoding='utf-8')

if class_map:
    class_map_path.write_text(json.dumps(class_map, ensure_ascii=False, indent=2), encoding='utf-8')
PY
