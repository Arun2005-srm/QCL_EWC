"""Inspect saved effective settings and epoch evidence without loading a model."""
import argparse
import json
from pathlib import Path
import yaml


def inspect_run(directory):
    directory = Path(directory)
    specs = yaml.safe_load((directory / "resolved.yaml").read_text(encoding="utf-8"))
    experiment = specs["experiment"]
    history_file = directory / "history.json"
    history = json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else []
    report = {"model": experiment.get("model", {}), "training": experiment.get("training", {}),
              "ewc": experiment.get("ewc", {}), "tasks": [],
              "completed_epoch_records": len(history), "last_epoch": history[-1] if history else None}
    for task in specs["tasks"]:
        d = task["dataset"]
        report["tasks"].append({"name": d["name"], "classes": d["classes"],
                                "labels": d["labels"], "transforms": d.get("transforms"),
                                "tiling": d.get("tiling"), "data_loader": task.get("data_loader")})
    if history:
        for split in ("training", "validation"):
            metric = history[-1].get(split, {})
            matrix = metric.get("confusion")
            if matrix:
                support = [sum(row) for row in matrix]
                predicted = [sum(row[c] for row in matrix) for c in range(len(matrix))]
                report[f"{split}_class_support"] = support
                report[f"{split}_predicted_pixels"] = predicted
                report[f"{split}_majority_class_accuracy"] = max(support) / sum(support) if sum(support) else None
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.run), indent=2))
