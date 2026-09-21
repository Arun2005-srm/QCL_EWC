"""Validate a dataset or a configured continual stream without loading SAM."""
import argparse
import json
from pathlib import Path

import yaml

from data import load_config, load_task_configs, build_records, build_dataloaders, save_manifest
from data.validation import validate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Single dataset YAML")
    source.add_argument("--experiment", help="YAML listing ordered dataset configs")
    parser.add_argument("--output", default="outputs/data_validation")
    parser.add_argument("--max-scenes", type=int, help="Partial scan; omit to validate every scene")
    args = parser.parse_args()
    if args.max_scenes is not None and args.max_scenes < 1:
        parser.error("--max-scenes must be positive")
    configs = load_task_configs(args.experiment) if args.experiment else [load_config(args.config)]
    for index, cfg in enumerate(configs):
        records = build_records(cfg["dataset"])
        report = validate_records(records, cfg["dataset"], args.max_scenes)
        loaders = build_dataloaders(cfg["dataset"], cfg.get("data_loader"), records=records)
        report["tiles"] = {k: len(v.dataset) for k, v in loaders.items()}
        batch = next(iter(loaders["train"]))
        report["batch"] = {"image": list(batch["image"].shape), "mask": list(batch["mask"].shape)}
        destination = Path(args.output) / f"task_{index:02d}"
        destination.mkdir(parents=True, exist_ok=True)
        save_manifest(records, destination / "splits.csv")
        (destination / "resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
