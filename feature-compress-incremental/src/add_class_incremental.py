# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data.coco_cls import build_dataset, read_class_map
from .data.transforms import build_transform
from .models.clip_wrapper import FrozenCLIPEncoder
from .models.compress_head import CompressionHead
from .models.classifier import LinearClassifier
from .models.ncfm_aligner import NCFMAligner
from .utils.common import (
    ensure_dir,
    load_yaml_config,
    get_device,
    set_seed,
    save_json,
    log_experiment_info,
    setup_logging,
)
from .utils.metrics import (
    topk_accuracy,
    build_confusion,
    plot_confusion,
    confusion_to_dict,
    per_class_accuracy,
    summarize_incremental_metrics,
)
from .utils.ncfm import CharacteristicFunctionLoss
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Incrementally add new classes.')
    parser.add_argument('--paths', type=Path, required=True, help='Path to the global paths config.')
    parser.add_argument('--task_cfg', type=Path, required=True, help='Path to the incremental task config.')
    parser.add_argument('--backbone', type=str, default='ViT-B-16', help='CLIP backbone name.')
    parser.add_argument('--pretrained', type=str, default='openai', help='Identifier or path to CLIP weights.')
    parser.add_argument('--resume_ckpt', type=Path, required=True, help='Checkpoint from the previous stage.')
    parser.add_argument('--n_compress', type=int, default=32, help='Compression head output dimension.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of fine-tuning epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--use_cache', action='store_true', help='Reuse cached features when available.')
    parser.add_argument('--save_dir', type=Path, required=True, help='Output directory for this stage.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--add_channels', type=int, default=0, help='Extra channels for the compression head.')
    parser.add_argument('--replay_per_class', type=int, default=5, help='Number of replay samples per old class.')
    parser.add_argument('--use_patch', action='store_true', help='Extract patch level features.')
    parser.add_argument('--dummy', action='store_true', help='Skip training and exit early (debug helper).')
    parser.add_argument('--use_ncfm', action='store_true', help='Enable NCFM-based feature alignment.')
    parser.add_argument('--ncfm_hidden', type=int, default=0, help='Hidden dimension for the NCFM aligner (0 = same as input).')
    parser.add_argument('--ncfm_freqs', type=int, default=128, help='Number of frequency probes for CF matching.')
    parser.add_argument('--ncfm_alpha', type=float, default=0.5, help='Weight for amplitude term in CF loss.')
    parser.add_argument('--ncfm_beta', type=float, default=0.5, help='Weight for phase term in CF loss.')
    parser.add_argument('--ncfm_weight', type=float, default=0.1, help='Overall weight for the CF matching loss.')
    parser.add_argument('--ncfm_batch', type=int, default=256, help='Number of cached old features sampled per step for CF loss.')
    return parser.parse_args()


def load_configs(args: argparse.Namespace) -> Dict:
    paths_cfg = load_yaml_config(args.paths)
    task_cfg = load_yaml_config(args.task_cfg)
    merged = {**paths_cfg, **task_cfg}
    merged['backbone'] = args.backbone
    merged['pretrained'] = args.pretrained
    merged['n_compress'] = args.n_compress
    merged['epochs'] = args.epochs
    merged['batch_size'] = args.batch_size
    merged['lr'] = args.lr
    merged['weight_decay'] = args.weight_decay
    merged['use_patch'] = args.use_patch
    merged['use_cache'] = args.use_cache
    merged['task_name'] = task_cfg.get('task_name', 'add_class')
    merged['previous_stage_ckpt'] = str(args.resume_ckpt)
    merged['use_ncfm'] = args.use_ncfm
    merged['ncfm_hidden'] = args.ncfm_hidden
    merged['ncfm_freqs'] = args.ncfm_freqs
    merged['ncfm_alpha'] = args.ncfm_alpha
    merged['ncfm_beta'] = args.ncfm_beta
    merged['ncfm_weight'] = args.ncfm_weight
    merged['ncfm_batch'] = args.ncfm_batch
    return merged


def sample_replay(
    features: torch.Tensor,
    labels: torch.Tensor,
    per_class: int,
    old_class_count: int,
) -> torch.Tensor:
    indices: List[torch.Tensor] = []
    for class_id in range(old_class_count):
        mask = (labels == class_id).nonzero(as_tuple=False).view(-1)
        if mask.numel() == 0:
            continue
        choice = mask[torch.randperm(mask.numel())[:per_class]]
        indices.append(choice)
    if not indices:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(indices, dim=0)


def _prepare_new_class_names(cfg: Dict) -> List[str]:
    names = list(cfg.get('new_class_names', ['new_cls']))
    ids = cfg.get('new_class_ids', [])
    expected = len(ids) if ids else len(names)
    if len(names) < expected:
        names.extend([f'new_cls_{i}' for i in range(len(names), expected)])
    else:
        names = names[:expected]
    return names


def _resolve_previous_class_names(
    checkpoint: Dict,
    cfg: Dict,
    sorted_items: List[tuple[str, int]],
    prev_output_dim: int,
) -> List[str]:
    prev_config = checkpoint.get('config') or {}
    ordered: List[str] = list(prev_config.get('class_names', []))
    if not ordered and 'class_names' in cfg:
        ordered = list(cfg['class_names'])
    if not ordered:
        ordered = [name for name, _ in sorted_items]
    if len(ordered) < prev_output_dim:
        fallback = [name for name, _ in sorted_items if name not in ordered]
        for name in fallback:
            ordered.append(name)
            if len(ordered) >= prev_output_dim:
                break
    while len(ordered) < prev_output_dim:
        ordered.append(f'class_{len(ordered)}')
    return ordered[:prev_output_dim]


def main() -> None:
    args = parse_args()
    cfg = load_configs(args)
    set_seed(args.seed)
    ensure_dir(args.save_dir)
    log_path = setup_logging(args.save_dir, 'add_class_incremental.log')
    logger.info(f'Log file: {log_path}')
    device = get_device()
    log_experiment_info(cfg)

    if args.dummy:
        logger.warning('Dummy mode is not implemented; exiting.')
        return

    image_size = cfg.get('image_size', 224)
    transform_train = build_transform(image_size, True)
    transform_val = build_transform(image_size, False)

    base_train_dataset = build_dataset(Path(cfg['train_list']), transform_train)
    base_val_dataset = build_dataset(Path(cfg['val_list']), transform_val)
    new_train_dataset = build_dataset(Path(cfg['new_train_list']), transform_train)
    new_val_dataset = build_dataset(Path(cfg['new_val_list']), transform_val)

    train_loader_base = DataLoader(
        base_train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader_base = DataLoader(
        base_val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader_new = DataLoader(
        new_train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader_new = DataLoader(
        new_val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_map = read_class_map(Path(cfg['class_map']))
    sorted_items = sorted(class_map.items(), key=lambda kv: kv[1])
    new_class_names = _prepare_new_class_names(cfg)

    cache_root = Path(cfg.get('cache_root', args.save_dir / 'cache'))
    encoder = FrozenCLIPEncoder(args.backbone, args.pretrained, device, cache_root)
    return_patch = cfg.get('use_patch', False)

    base_cache = encoder.extract_and_cache(
        train_loader_base,
        'train',
        image_size,
        return_patch,
        args.use_cache,
    )
    new_cache = encoder.extract_and_cache(
        train_loader_new,
        'train_new',
        image_size,
        return_patch,
        False,
    )
    val_base_cache = encoder.extract_and_cache(
        val_loader_base,
        'val',
        image_size,
        return_patch,
        args.use_cache,
    )
    val_new_cache = encoder.extract_and_cache(
        val_loader_new,
        'val_new',
        image_size,
        return_patch,
        False,
    )

    checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
    prev_cfg = checkpoint.get('config', {})
    if prev_cfg.get('use_ncfm', False) and not args.use_ncfm:
        logger.warning('Previous stage used NCFM; enabling it for compatibility.')
        args.use_ncfm = True
        cfg['use_ncfm'] = True
        for key in ('ncfm_hidden', 'ncfm_freqs', 'ncfm_alpha', 'ncfm_beta', 'ncfm_weight', 'ncfm_batch'):
            if key in prev_cfg:
                setattr(args, key, prev_cfg[key])
                cfg[key] = prev_cfg[key]
    input_dim = encoder.output_dim(return_patch)
    prev_output_dim = checkpoint['classifier']['head.weight'].shape[0]
    old_class_names = _resolve_previous_class_names(
        checkpoint,
        cfg,
        sorted_items,
        prev_output_dim,
    )

    class_names = old_class_names + new_class_names
    cfg['class_names'] = class_names
    cfg['total_classes'] = len(class_names)

    replay_indices = sample_replay(
        base_cache['features'],
        base_cache['labels'],
        args.replay_per_class,
        prev_output_dim,
    )
    replay_features = (
        base_cache['features'][replay_indices]
        if replay_indices.numel() > 0
        else torch.empty(0)
    )
    replay_labels = (
        base_cache['labels'][replay_indices]
        if replay_indices.numel() > 0
        else torch.empty(0, dtype=torch.long)
    )

    new_labels = torch.full(
        (new_cache['features'].shape[0],),
        prev_output_dim,
        dtype=torch.long,
    )
    if replay_features.numel() > 0:
        combined_features = torch.cat([new_cache['features'], replay_features], dim=0)
        combined_labels = torch.cat([new_labels, replay_labels], dim=0)
    else:
        combined_features = new_cache['features']
        combined_labels = new_labels

    train_dataset = TensorDataset(combined_features.float(), combined_labels.long())
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_features = torch.cat(
        [val_base_cache['features'], val_new_cache['features']],
        dim=0,
    )
    val_labels = torch.cat(
        [
            val_base_cache['labels'],
            torch.full(
                (val_new_cache['features'].shape[0],),
                prev_output_dim,
                dtype=torch.long,
            ),
        ],
        dim=0,
    )
    val_dataset = TensorDataset(val_features.float(), val_labels.long())
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    compressor = CompressionHead(input_dim, cfg['n_compress'], return_patch)
    classifier = LinearClassifier(compressor.output_dim(), prev_output_dim)
    compressor.load_state_dict(checkpoint['compressor'])
    classifier.load_state_dict(checkpoint['classifier'])
    if args.add_channels:
        compressor.add_channels(args.add_channels)
        cfg['n_compress'] = compressor.output_dim()

    if args.use_ncfm:
        hidden_dim = args.ncfm_hidden if args.ncfm_hidden > 0 else None
        aligner = NCFMAligner(compressor.output_dim(), hidden_dim)
        if 'aligner' in checkpoint:
            try:
                aligner.load_state_dict(checkpoint['aligner'])
            except RuntimeError as err:
                logger.warning(f'Failed to load aligner weights ({err}); reinitializing.')
    else:
        aligner = nn.Identity()

    if new_class_names:
        classifier.expand_classes(len(new_class_names))
    compressor.to(device)
    classifier.to(device)
    aligner = aligner.to(device)

    params = list(compressor.parameters()) + list(classifier.parameters())
    if args.use_ncfm:
        params += list(aligner.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    cf_loss_fn = CharacteristicFunctionLoss(
        num_freqs=args.ncfm_freqs,
        alpha=args.ncfm_alpha,
        beta=args.ncfm_beta,
    ) if args.use_ncfm and args.ncfm_weight > 0 else None

    base_features_cache = base_cache['features'].float()

    best_top1 = 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        compressor.train()
        classifier.train()
        aligner.train()
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            compressed = compressor(feats)
            aligned = aligner(compressed)
            logits = classifier(aligned)
            loss = criterion(logits, labels)

            if cf_loss_fn is not None:
                old_mask = labels < prev_output_dim
                num_old = int(old_mask.sum().item())
                if num_old > 0:
                    sample_size = min(num_old, args.ncfm_batch)
                    student_feats = aligned[old_mask][:sample_size]
                    base_indices = torch.randint(
                        0,
                        base_features_cache.shape[0],
                        (sample_size,),
                        device='cpu',
                    )
                    target_batch = base_features_cache[base_indices].to(device)
                    with torch.no_grad():
                        target_compressed = compressor(target_batch)
                        target_aligned = aligner(target_compressed)
                    cf_loss = cf_loss_fn(target_aligned.detach(), student_feats)
                    loss = loss + args.ncfm_weight * cf_loss

            loss.backward()
            optimizer.step()

        compressor.eval()
        classifier.eval()
        aligner.eval()
        logits_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                logits = classifier(aligner(compressor(feats)))
                logits_list.append(logits.cpu())
                labels_list.append(labels)
        logits_tensor = torch.cat(logits_list, dim=0)
        labels_tensor = torch.cat(labels_list, dim=0)

        acc = topk_accuracy(logits_tensor, labels_tensor, ks=(1, 5))
        logger.info(f'Epoch {epoch}: top1={acc[1]:.4f} top5={acc[5]:.4f}')

        if acc[1] > best_top1:
            best_top1 = acc[1]
            best_state = {
                'compressor': compressor.state_dict(),
                'classifier': classifier.state_dict(),
                'config': cfg,
            }
            if args.use_ncfm:
                best_state['aligner'] = aligner.state_dict()
            torch.save(best_state, args.save_dir / 'best_incremental.ckpt')

    last_state = {
        'compressor': compressor.state_dict(),
        'classifier': classifier.state_dict(),
        'config': cfg,
    }
    if args.use_ncfm:
        last_state['aligner'] = aligner.state_dict()
    torch.save(last_state, args.save_dir / 'last_incremental.ckpt')

    confusion = build_confusion(logits_tensor, labels_tensor, len(class_names))
    plot_confusion(confusion, class_names, args.save_dir / 'confusion_incremental.png')
    save_json(
        confusion_to_dict(confusion, class_names),
        args.save_dir / 'confusion_incremental.json',
    )
    per_class_acc = per_class_accuracy(confusion, class_names)
    save_json(per_class_acc, args.save_dir / 'per_class_accuracy.json')

    summary_metrics = summarize_incremental_metrics(
        confusion,
        class_names,
        old_class_names,
        baseline_old_acc=cfg.get('baseline_old_acc'),
        baseline_new_acc=cfg.get('baseline_new_acc'),
    )
    summary_metrics['best_top1'] = best_top1
    summary_metrics['replay_per_class'] = args.replay_per_class
    save_json(summary_metrics, args.save_dir / 'metrics_incremental.json')


if __name__ == '__main__':
    main()
