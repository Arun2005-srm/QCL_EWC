"""Evaluate saved alignment weights against held-out configured datasets."""
import argparse
import json
from pathlib import Path

import torch

from continual.experiment import file_hash
from continual.metrics import evaluate
from data import build_dataloaders
from models.sam_segmenter import UniversalSAM
from continual.visualization import save_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Alignment training checkpoint")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--num-visualizations", type=int, default=3)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    specs = state["specs"]
    if file_hash(args.sam_checkpoint) != specs["sam_sha256"]:
        parser.error("SAM checkpoint does not match the training checkpoint")
    tasks = specs["tasks"]
    classes = tasks[0]["dataset"]["classes"]
    model = UniversalSAM(len(classes), args.sam_checkpoint, **specs["experiment"].get("model", {})).to(args.device)
    model.load_trainable_state(state["model"])
    from data.contracts import SampleRecord
    report = {}
    # Reuse the exact saved partitions, rather than rediscovering/resplitting.
    seen_tasks = min(len(tasks), state["task"] + (1 if state["epoch"] > 0 else 0))
    for index, (cfg, saved) in enumerate(zip(tasks[:seen_tasks], state["records"][:seen_tasks])):
        records = {s: [SampleRecord(i, Path(x), Path(y), g, s) for i, x, y, g in rows] for s, rows in saved.items()}
        loader = build_dataloaders(cfg["dataset"], cfg.get("data_loader"), records=records)["test"]
        report[cfg["dataset"]["name"]] = evaluate(model, loader, args.device, len(classes), cfg["dataset"]["labels"].get("ignore_index", 255))
        if args.num_visualizations > 0:
            save_predictions(model, loader.dataset, args.device, Path(args.output).parent / f"task_{index:02d}_predictions", args.num_visualizations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
