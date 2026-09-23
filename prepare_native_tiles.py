"""Recover original rasters from preprocessing provenance and export native tiles.

Existing prepared config determines scene splits BEFORE tiling. Original rasters
are decoded once per scene; their pixels are never resized. Outputs are separate.
"""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from tqdm.auto import tqdm

from data import load_config, build_records
from prepare_datasets import Pair, load_pair, output_id, _save_png


def prepare(config, provenance, output, size=512, exclude_regions=()):
    if size < 32:
        raise ValueError("Tile size must be at least 32")
    cfg = load_config(config)
    dataset = cfg["dataset"]
    if dataset["labels"].get("encoding", "index") != "index":
        raise ValueError("Native export currently requires index masks")
    mapping = dataset["labels"]["mapping"]
    if dataset["labels"].get("ignore_index", 255) != 255:
        raise ValueError("Native PNG export requires ignore_index=255")
    sources = {}
    for line in Path(provenance).read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry["id"] in sources:
            raise ValueError("Duplicate source provenance ID")
        sources[entry["id"]] = entry
    records = build_records(dataset)
    selected, excluded = [], []
    for split, subset in records.items():
        for record in subset:
            source = sources[record.sample_id]
            if source["source_id"].split("/")[0] in exclude_regions:
                excluded.append(source["source_id"])
                continue
            for key in ("source_image", "source_mask"):
                if not Path(source[key]).is_file():
                    raise FileNotFoundError(source[key])
            selected.append((split, record, source))
    if any(not any(s == split for s, _, _ in selected) for split in ("train", "val", "test")):
        raise ValueError("Exclusions leave an empty split")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    for folder in ("images", "masks"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    counts = {s: 0 for s in records}
    scene_counts = {s: 0 for s in records}
    with (output / "pairs.csv").open("w", newline="", encoding="utf-8") as manifest, \
            (output / "tiles.jsonl").open("w", encoding="utf-8") as provenance_out:
        writer = csv.DictWriter(manifest, fieldnames=["id", "image", "mask", "group", "split"])
        writer.writeheader()
        for split, record, source in tqdm(selected, desc=dataset["name"], unit="scene"):
            pair = Pair(source["source_id"], Path(source["source_image"]),
                        Path(source["source_mask"]), record.group, split)
            image, raw_mask, _ = load_pair(pair, mapping)
            # Persist harmonized IDs once, with an identity mapping in output YAML.
            mask = np.full(raw_mask.shape, 255, dtype=np.uint8)
            for raw, mapped in mapping.items():
                mask[raw_mask == raw] = mapped
            h, w = mask.shape
            scene_counts[split] += 1
            for top in range(0, h, size):
                for left in range(0, w, size):
                    identifier = output_id(f"{record.sample_id}@{top},{left}")
                    rgb = Image.new("RGB", (size, size))
                    labels = Image.new("L", (size, size), 255)
                    rgb.paste(Image.fromarray(image[top:top + size, left:left + size]), (0, 0))
                    labels.paste(Image.fromarray(mask[top:top + size, left:left + size]), (0, 0))
                    _save_png(rgb, output / "images" / f"{identifier}.png")
                    _save_png(labels, output / "masks" / f"{identifier}.png")
                    writer.writerow(dict(id=identifier, image=f"images/{identifier}.png",
                                         mask=f"masks/{identifier}.png", group=record.group, split=split))
                    provenance_out.write(json.dumps({**source, "id": identifier,
                                                     "prepared_scene_id": record.sample_id,
                                                     "top": top, "left": left, "size": size}) + "\n")
                    counts[split] += 1
    cfg = deepcopy(cfg)
    cfg.pop("defaults", None)
    cfg["dataset"].update(manifest=str(output / "pairs.csv"), split={"strategy": "manifest"},
                          tiling={"enabled": False})
    cfg["dataset"]["labels"]["mapping"] = {i: i for i in range(len(dataset["classes"]))} | {255: 255}
    cfg["dataset"].setdefault("transforms", {}).update(image_size=[size, size], normalization=None)
    (output / "dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    report = dict(scene_counts=scene_counts, tile_counts=counts, excluded=excluded,
                  source_config=str(Path(config).resolve()), tile_size=size,
                  cross_dataset_overlap_audited=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sources", required=True, help="Prepared dataset sources.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--exclude-regions", nargs="*", default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.sources, args.output, args.size, args.exclude_regions), indent=2))
