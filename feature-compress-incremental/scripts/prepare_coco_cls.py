#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from pycocotools.coco import COCO


def assign_label_map(cat_ids, label_map):
    next_index = len(label_map)
    for cid in cat_ids:
        key = str(int(cid))
        if key not in label_map:
            label_map[key] = next_index
            next_index += 1
    return label_map


def build_records(coco: COCO, image_root: Path, candidate_ids, label_map):
    if not candidate_ids:
        return []
    cat_ids = list({int(cid) for cid in candidate_ids})
    img_ids = set()
    for cid in cat_ids:
        img_ids.update(coco.getImgIds(catIds=[cid]))
    records = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        areas = defaultdict(float)
        for ann in anns:
            areas[ann['category_id']] += float(ann.get('area', 0.0))
        if not areas:
            continue
        main_cid, _ = max(areas.items(), key=lambda kv: kv[1])
        key = str(main_cid)
        if key not in label_map:
            continue
        file_name = coco.imgs[img_id]['file_name']
        img_path = image_root / file_name
        records.append((str(img_path), label_map[key]))
    return records


def run(paths_cfg: Path, task_cfg: Path) -> None:
    paths = yaml.safe_load(paths_cfg.read_text(encoding='utf-8'))
    task = yaml.safe_load(task_cfg.read_text(encoding='utf-8'))

    train_list_path = Path(paths['train_list']).expanduser()
    val_list_path = Path(paths['val_list']).expanduser()
    new_train_path = Path(paths.get('new_train_list', train_list_path.parent / 'new_train_list.txt')).expanduser()
    new_val_path = Path(paths.get('new_val_list', val_list_path.parent / 'new_val_list.txt')).expanduser()
    class_map_path = Path(paths['class_map']).expanduser()

    for p in (train_list_path, val_list_path, new_train_path, new_val_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    cache_root = Path(paths.get('cache_root', train_list_path.parent)).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    if class_map_path.exists():
        try:
            class_map = json.loads(class_map_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            class_map = {}
    else:
        class_map = {}

    image_root = Path(paths['coco_root']).expanduser()
    annotation_file = Path(paths['annotation_file']).expanduser()
    coco = COCO(str(annotation_file))

    base_class_ids = task.get('class_ids') or []
    new_class_ids = task.get('new_class_ids') or []
    random_seed = task.get('split_seed', 42)
    train_ratio = task.get('train_ratio', 0.85)
    random.seed(random_seed)

    if base_class_ids:
        class_map = assign_label_map(base_class_ids, {})
        records = build_records(coco, image_root, base_class_ids, class_map)
        if records:
            random.shuffle(records)
            split = int(len(records) * train_ratio)
            train_records = records[:split]
            val_records = records[split:]
            train_list_path.write_text('\n'.join(f"{p} {l}" for p, l in train_records), encoding='utf-8')
            val_list_path.write_text('\n'.join(f"{p} {l}" for p, l in val_records), encoding='utf-8')
        else:
            print('[WARN] 未找到任何 base 类别图像')
            train_list_path.write_text('', encoding='utf-8')
            val_list_path.write_text('', encoding='utf-8')

    if new_class_ids:
        class_map = assign_label_map(new_class_ids, class_map)
        records_new = build_records(coco, image_root, new_class_ids, class_map)
        if records_new:
            random.shuffle(records_new)
            split_new = int(len(records_new) * train_ratio)
            new_train = records_new[:split_new]
            new_val = records_new[split_new:]
            new_train_path.write_text('\n'.join(f"{p} {l}" for p, l in new_train), encoding='utf-8')
            new_val_path.write_text('\n'.join(f"{p} {l}" for p, l in new_val), encoding='utf-8')
            print(f"[INFO] 写入 {len(new_train)} 新训练样本, {len(new_val)} 新验证样本")
        else:
            print('[WARN] 未找到任何增量类别图像')
            new_train_path.write_text('', encoding='utf-8')
            new_val_path.write_text('', encoding='utf-8')

    if class_map:
        class_map_path.write_text(json.dumps(class_map, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"[INFO] 类别映射共 {len(class_map)} 类，已写入 {class_map_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare COCO classification splits')
    parser.add_argument('--paths', required=True, type=Path, help='路径配置 YAML 文件')
    parser.add_argument('--task', required=True, type=Path, help='任务配置 YAML 文件')
    args = parser.parse_args()
    run(args.paths, args.task)


if __name__ == '__main__':
    main()
