import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SegmentationDataset
from .discovery import discover_pairs
from .splitting import split_records


def build_records(dataset_cfg):
    return split_records(discover_pairs(dataset_cfg), dataset_cfg.get("split", {}))


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders(dataset_cfg, loader_cfg=None, *, records=None):
    """Same two-config interface as the reference project; train/val/test loaders."""
    loader_cfg = loader_cfg or {}
    records = build_records(dataset_cfg) if records is None else records
    workers = loader_cfg.get("num_workers", 0)
    seed = loader_cfg.get("seed", dataset_cfg.get("split", {}).get("seed", 42))
    # Also seed transforms executed in the main process (num_workers=0).
    torch.manual_seed(seed)
    loaders = {}
    for i, (split, subset) in enumerate(records.items()):
        dataset = SegmentationDataset(subset, dataset_cfg, training=split == "train")
        generator = torch.Generator().manual_seed(seed + i)
        loaders[split] = DataLoader(
            dataset, batch_size=loader_cfg.get("batch_size", 1),
            shuffle=split == "train" and len(dataset) > 0,
            num_workers=workers, pin_memory=loader_cfg.get("pin_memory", False),
            persistent_workers=workers > 0 and loader_cfg.get("persistent_workers", False),
            worker_init_fn=seed_worker, generator=generator, drop_last=False,
        )
    return loaders


def save_manifest(records, destination):
    """Save scene splits BEFORE tiling; reusable through dataset.manifest."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image", "mask", "group", "split"])
        writer.writeheader()
        for split, subset in records.items():
            for r in subset:
                writer.writerow({"id": r.sample_id, "image": str(r.image_path), "mask": str(r.mask_path), "group": r.group, "split": split})
