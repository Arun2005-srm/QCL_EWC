"""Offline end-to-end data smoke test; no SAM weights or real data required."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch
import yaml

from data import load_config, build_records, build_dataloaders, save_manifest
from data.validation import validate_records


def make_fixture(root):
    root = Path(root)
    for folder in ("images", "masks"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for index in range(9):
        y, x = np.indices((40, 48))
        mask = ((x // 10 + y // 10 + index) % 5).astype(np.uint8)
        image = np.stack([mask * 40, (x * 5).astype(np.uint8), (y * 6).astype(np.uint8)], axis=-1)
        Image.fromarray(image).save(root / "images" / f"scene_{index}.png")
        Image.fromarray(mask).save(root / "masks" / f"scene_{index}_mask.png")
    cfg = {"dataset": {"name": "synthetic", "images": {"root": "images"},
                          "masks": {"root": "masks"}, "pairing": {"mask_suffix": "_mask"},
                          "classes": ["other", "building", "tree_woodland", "water", "road"],
                          "labels": {"encoding": "index", "mapping": {i: i for i in range(5)}, "ignore_index": 255},
                          "split": {"strategy": "group", "seed": 42, "ratios": {"train": .6, "val": .2, "test": .2}},
                          "tiling": {"enabled": True, "size": [32, 32], "stride": [32, 32]},
                          "transforms": {"image_size": [32, 32]}},
           "data_loader": {"batch_size": 2, "num_workers": 0}}
    path = root / "dataset.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def run(root, device, workers):
    path = make_fixture(root)
    cfg = load_config(path)
    cfg["data_loader"]["num_workers"] = workers
    records = build_records(cfg["dataset"])
    report = validate_records(records, cfg["dataset"])
    save_manifest(records, Path(root) / "splits.csv")
    loaders = build_dataloaders(cfg["dataset"], cfg["data_loader"], records=records)
    model = torch.nn.Conv2d(3, 5, kernel_size=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    batch = next(iter(loaders["train"]))
    logits = model(batch["image"].to(device))
    loss = torch.nn.functional.cross_entropy(logits, batch["mask"].to(device), ignore_index=255)
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss) and model.weight.grad.abs().sum() > 0
    optimizer.step()
    for split in ("val", "test"):
        assert next(iter(loaders[split]))["mask"].dtype == torch.long
    report.update(status="passed", test="data pipeline + toy segmentation backward (not SAM)",
                  device=str(device), workers=workers, loss=float(loss.detach().cpu()),
                  tiles={k: len(v.dataset) for k, v in loaders.items()},
                  image_shape=list(batch["image"].shape), mask_shape=list(batch["mask"].shape),
                  torch_version=torch.__version__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="outputs/smoke.json")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; install a CUDA-enabled PyTorch build on the DGX")
    with tempfile.TemporaryDirectory(prefix="qcl_smoke_") as root:
        report = run(root, torch.device(args.device), args.num_workers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
