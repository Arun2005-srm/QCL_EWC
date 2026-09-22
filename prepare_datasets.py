"""Organize paired rasters and resize RGB images/index masks for SAM.

Standalone CPU preprocessing: does not import torch, SAM, or PennyLane.
Source files are never modified. Raw mask IDs are preserved on disk.
"""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import tifffile
import yaml

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CLASSES = ["other", "building", "tree_woodland", "water", "road"]
SPLIT_NAMES = {"train", "val", "test"}


@dataclass(frozen=True)
class Pair:
    identifier: str
    image: Path
    mask: Path
    group: str
    split: str = ""


def resolve(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def _id(path, root, aliases, suffix):
    parts = list(path.relative_to(root).parts)
    # Remove just the nearest image/mask folder, retaining region/split paths.
    positions = [i for i, part in enumerate(parts[:-1]) if part.lower() in aliases]
    if not positions:
        return None
    del parts[positions[-1]]
    stem = Path(parts[-1]).stem
    if suffix:
        if not stem.endswith(suffix):
            raise ValueError(f"Expected filename suffix {suffix!r}: {path}")
        stem = stem[:-len(suffix)]
    return "/".join([*parts[:-1], stem])


def discover(spec, parent, allow_unlabeled=False, paired_only=False, orphan_masks_out=None):
    root = resolve(spec["source"], parent)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset source does not exist: {root}")
    skipped = []
    if spec.get("source_manifest"):
        manifest = resolve(spec["source_manifest"], parent)
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"id", "image", "mask"}.issubset(reader.fieldnames or []):
                raise ValueError("source_manifest requires id,image,mask columns")
            pairs = []
            for row in reader:
                if not all((row.get(k) or "").strip() for k in ("id", "image", "mask")):
                    raise ValueError("Empty id/image/mask in source manifest")
                pairs.append(Pair(row["id"].strip(), resolve(row["image"], manifest.parent),
                                  resolve(row["mask"], manifest.parent),
                                  (row.get("group") or row["id"]).strip(), (row.get("split") or "").strip()))
    else:
        image_aliases = {s.lower() for s in spec.get("image_dirs", ["images", "imgs"])}
        mask_aliases = {s.lower() for s in spec.get("mask_dirs", ["masks", "labels"])}
        if image_aliases & mask_aliases:
            raise ValueError("Image and mask folder names must be disjoint")
        images, masks, unclassified = {}, {}, []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            image_id = _id(path, root, image_aliases, spec.get("image_suffix", ""))
            mask_id = _id(path, root, mask_aliases, spec.get("mask_suffix", ""))
            if image_id is not None and mask_id is not None:
                raise ValueError(f"Ambiguous image/mask folder path: {path}; use a source_manifest")
            if image_id is None and mask_id is None:
                unclassified.append(str(path))
                continue
            table, identifier = (images, image_id) if image_id is not None else (masks, mask_id)
            if identifier in table:
                raise ValueError(f"Duplicate pair ID {identifier}: {table[identifier]} and {path}")
            table[identifier] = path.resolve()
        if unclassified:
            raise ValueError(f"Cannot classify {len(unclassified)} raster files, e.g. {unclassified[:3]}. "
                             "Edit image_dirs/mask_dirs or provide a source_manifest to select pairs explicitly.")
        missing_masks = sorted(images.keys() - masks.keys())
        orphan_masks = sorted(masks.keys() - images.keys())
        if not paired_only and (orphan_masks or (missing_masks and not allow_unlabeled)):
            raise ValueError(f"Unpaired files: images without masks={missing_masks[:8]}, masks without images={orphan_masks[:8]}. "
                             "Check suffixes/layout. --allow-unlabeled-images skips image-only samples; "
                             "--paired-only explicitly excludes both kinds of unmatched file and records them.")
        skipped = [str(images[i]) for i in missing_masks]
        if orphan_masks_out is not None:
            orphan_masks_out.extend(str(masks[i]) for i in orphan_masks)
        if paired_only and (missing_masks or orphan_masks):
            print(f"Pair intersection: retaining {len(images.keys() & masks.keys())} complete pairs; "
                  f"excluding {len(missing_masks)} images without masks and {len(orphan_masks)} masks without images. "
                  "Original files remain unchanged.", flush=True)
        pairs = []
        for identifier in sorted(images.keys() & masks.keys()):
            group = identifier
            if spec.get("group_regex"):
                match = re.search(spec["group_regex"], identifier)
                if not match or not match.groupdict().get("group"):
                    raise ValueError(f"group_regex must capture named 'group' for {identifier}")
                group = match.group("group")
            pairs.append(Pair(identifier, images[identifier], masks[identifier], group))
    if not pairs:
        raise ValueError(f"No paired images and masks discovered in {root}; archives must be extracted first")
    for field in ("identifier", "image", "mask"):
        values = [getattr(p, field) for p in pairs]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {field} in source pairs")
    has_splits = any(p.split for p in pairs)
    group_splits = {}
    for p in pairs:
        if not p.group:
            raise ValueError(f"Empty scene group: {p.identifier}")
        if has_splits and p.split not in SPLIT_NAMES:
            raise ValueError(f"Every source-manifest row must specify train/val/test if any row has a split: {p.identifier}")
        if p.group in group_splits and group_splits[p.group] != p.split:
            raise ValueError(f"Source group crosses official partitions: {p.group}")
        group_splits[p.group] = p.split
        for path in (p.image, p.mask):
            if not path.is_file():
                raise FileNotFoundError(path)
    return sorted(pairs, key=lambda p: p.identifier), skipped


def read_raster(path):
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            array, axes = series.asarray(), series.axes
        if axes in {"SYX", "CYX"}:
            array = array.transpose(1, 2, 0)
        elif axes not in {"YX", "YXS", "YXC"}:
            raise ValueError(f"Unsupported TIFF axes {axes}: {path}")
        return array
    with Image.open(path) as image:
        # Keep palette indices in masks; color masks are not guessed.
        return np.array(image)


def apply_split_files(pairs, spec, parent, skipped=()):
    """Apply scene-level lists. Never assign a whole scene using one of its tiles."""
    if not spec.get("split_files"):
        return pairs
    if spec.get("source_manifest"):
        raise ValueError("Choose source_manifest OR split_files, not both")
    root = resolve(spec["source"], parent)
    split_files = spec["split_files"]
    if set(split_files) != SPLIT_NAMES:
        raise ValueError("split_files must define train, val and test")
    def aliases(path, identifier=None):
        values = {path.name, path.stem, path.as_posix(), path.with_suffix("").as_posix()}
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            values.update([relative.as_posix(), relative.with_suffix("").as_posix()])
        if identifier:
            values.add(identifier)
        return values
    lookup = {}
    for index, pair in enumerate(pairs):
        for alias in aliases(pair.image, pair.identifier):
            lookup.setdefault(alias, set()).add(index)
    omitted = set()
    for path in skipped:
        omitted.update(aliases(Path(path)))
    assignment = {}
    for split in ("train", "val", "test"):
        path = resolve(split_files[split], root)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            token = line.strip().strip('"').replace("\\", "/")
            if not token or token.startswith("#"):
                continue
            while token.startswith("./"):
                token = token[2:]
            matches = lookup.get(token, set())
            if not matches and token in omitted:
                continue
            if len(matches) != 1:
                stem = Path(token).stem
                tile_parents = [p.identifier for p in pairs if stem.startswith(p.image.stem + "_")]
                if not matches and tile_parents:
                    raise ValueError(f"Split entry {token!r} in {path.name} appears to name a TILE of scene {tile_parents[0]!r}, "
                                     "but inputs are WHOLE scenes. These split lists cannot be applied to whole-scene resizing. "
                                     "Provide scene-level split lists/manifest, or explicitly use --split-policy scene for new scene-safe 70/15/15 splits.")
                raise ValueError(f"Split entry {token!r} in {path} matched {len(matches)} source images. "
                                 "Use matching scene IDs/paths, or --split-policy scene for explicitly new scene-level splits.")
            index = next(iter(matches))
            if index in assignment and assignment[index] != split:
                raise ValueError(f"Source image assigned to multiple splits: {pairs[index].identifier}")
            assignment[index] = split
    missing = [p.identifier for index, p in enumerate(pairs) if index not in assignment]
    if missing:
        raise ValueError(f"Split lists leave {len(missing)} paired scenes unassigned, e.g. {missing[:5]}")
    output = [replace(p, split=assignment[i]) for i, p in enumerate(pairs)]
    groups = {}
    for pair in output:
        if pair.group in groups and groups[pair.group] != pair.split:
            raise ValueError(f"Source group crosses split-file partitions: {pair.group}")
        groups[pair.group] = pair.split
    return output


def load_pair(pair, allowed_labels):
    image, mask = read_raster(pair.image), read_raster(pair.mask)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 RGB image, got {image.dtype} {image.shape}: {pair.image}. "
                         "This preparation profile requires explicit conversion for multispectral/16-bit inputs.")
    # Some indexed masks are saved as three identical grayscale channels.
    if mask.ndim == 3 and mask.shape[2] == 3 and np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 0], mask[..., 2]):
        mask = mask[..., 0]
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"Expected indexed integer mask: {pair.mask}. RGB color masks need an explicit color-to-ID conversion.")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image/mask dimensions differ: {pair.identifier}: {image.shape[:2]} versus {mask.shape}")
    values = set(np.unique(mask).tolist())
    if values - set(allowed_labels):
        raise ValueError(f"Unexpected raw mask IDs {sorted(values - set(allowed_labels))} in {pair.mask}; verify the release's labels")
    if min(values) < 0 or max(values) > 255:
        raise ValueError(f"This PNG export expects raw mask IDs between 0 and 255: {pair.mask}")
    return image, mask.astype(np.uint8), sorted(values)


def resize_pair(image, mask, size, mode):
    height, width = size
    old_h, old_w = mask.shape
    if mode == "stretch":
        new_h, new_w = height, width
    elif mode == "letterbox":
        scale = min(height / old_h, width / old_w)
        new_h, new_w = max(1, round(old_h * scale)), max(1, round(old_w * scale))
    else:
        raise ValueError("resize_mode must be stretch or letterbox")
    rgb = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)
    labels = Image.fromarray(mask).resize((new_w, new_h), Image.Resampling.NEAREST)
    if mode == "letterbox":
        padded_rgb = Image.new("RGB", (width, height), 0)
        padded_labels = Image.new("L", (width, height), 255)
        padded_rgb.paste(rgb, (0, 0))
        padded_labels.paste(labels, (0, 0))
        rgb, labels = padded_rgb, padded_labels
    return rgb, labels, [new_h, new_w]


def output_id(identifier):
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", identifier).strip("_")[:100] or "sample"
    return f"{stem}__{hashlib.sha256(identifier.encode()).hexdigest()[:12]}"


def _save_png(image, destination):
    temporary = destination.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(destination)


def prepare_dataset(name, spec, parent, output_root, size, mode, dry_run=False, limit=None,
                    overwrite=False, allow_unlabeled=False, paired_only=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Dataset names may contain only letters, numbers, underscores and hyphens")
    source = resolve(spec["source"], parent)
    destination = (output_root / name).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination trees must be separate")
    orphan_masks = []
    pairs, skipped = discover(spec, parent, allow_unlabeled, paired_only, orphan_masks)
    pairs = apply_split_files(pairs, spec, parent, [*skipped, *orphan_masks])
    selected = pairs[:limit] if limit else pairs
    for p in selected:
        if p.image.is_relative_to(destination) or p.mask.is_relative_to(destination):
            raise ValueError("Destination contains source-manifest input files")
    if not dry_run and destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Output exists: {destination}. Use a new output root or --overwrite to regenerate it")
    mapping = spec.get("label_mapping", {})
    if not mapping or any(type(k) is not int or type(v) is not int or v not in {0, 1, 2, 3, 4, 255} for k, v in mapping.items()):
        raise ValueError("label_mapping must map integer source IDs into the shared five classes or 255")
    if mode == "letterbox" and mapping.get(255) != 255:
        raise ValueError("Letterbox padding requires label_mapping: {255: 255, ...}")
    ids = [output_id(p.identifier) for p in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("Output filename collision")
    if not dry_run:
        for folder in ("images", "masks"):
            (destination / folder).mkdir(parents=True, exist_ok=True)
    rows, sources, label_counts = [], [], Counter()
    for index, (pair, identifier) in enumerate(zip(selected, ids), 1):
        image, mask, raw_ids = load_pair(pair, mapping)
        label_counts.update(raw_ids)
        if not dry_run:
            rgb, labels, valid_size = resize_pair(image, mask, size, mode)
            output_labels = set(np.unique(np.asarray(labels)).tolist())
            if not output_labels.issubset(set(raw_ids) | ({255} if mode == "letterbox" else set())):
                raise AssertionError("Mask resizing introduced new class IDs")
            _save_png(rgb, destination / "images" / f"{identifier}.png")
            _save_png(labels, destination / "masks" / f"{identifier}.png")
            # Verify persisted dimensions and IDs, not just in-memory arrays.
            with Image.open(destination / "images" / f"{identifier}.png") as check:
                if check.size != (size[1], size[0]) or check.mode != "RGB":
                    raise AssertionError("Image export verification failed")
            with Image.open(destination / "masks" / f"{identifier}.png") as check:
                if check.size != (size[1], size[0]) or not np.array_equal(np.asarray(check), np.asarray(labels)):
                    raise AssertionError("Mask export verification failed")
            rows.append({"id": identifier, "image": f"images/{identifier}.png", "mask": f"masks/{identifier}.png", "group": pair.group, "split": pair.split})
            sources.append({"id": identifier, "source_id": pair.identifier, "source_image": str(pair.image), "source_mask": str(pair.mask),
                            "original_size": list(mask.shape), "resized_content_size": valid_size, "raw_labels": raw_ids})
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"{name}: {'checked' if dry_run else 'saved'} {index}/{len(selected)} pairs", flush=True)
    report = {"dataset": name, "source": str(source), "destination": str(destination), "dry_run": dry_run,
              "discovered_pairs": len(pairs), "processed_pairs": len(selected), "size": size, "resize_mode": mode,
              "raw_label_scene_counts": dict(sorted(label_counts.items())), "skipped_unlabeled_images": skipped,
              "skipped_orphan_masks": orphan_masks, "paired_only": paired_only,
              "limited_run": len(selected) < len(pairs), "official_splits_supplied": bool(pairs[0].split),
              "source_split_counts": dict(Counter(p.split or "unassigned" for p in pairs))}
    if not dry_run:
        with (destination / "pairs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "image", "mask", "group", "split"])
            writer.writeheader()
            writer.writerows(rows)
        with (destination / "sources.jsonl").open("w", encoding="utf-8") as handle:
            for item in sources:
                handle.write(json.dumps(item) + "\n")
        split = {"strategy": "manifest"} if pairs[0].split else {"strategy": "group", "seed": 42, "ratios": {"train": .7, "val": .15, "test": .15}}
        cfg = {"data_loader": {"batch_size": 1, "num_workers": 0, "pin_memory": True, "seed": 42},
               "dataset": {"name": name, "manifest": str(destination / "pairs.csv"),
                           "images": {"root": str(destination / "images"), "extensions": [".png"], "scale": 255.0},
                           "masks": {"root": str(destination / "masks"), "extensions": [".png"]},
                           "classes": CLASSES, "labels": {"encoding": "index", "ignore_index": 255, "mapping": mapping},
                           "split": split, "tiling": {"enabled": False},
                           "transforms": {"image_size": size, "normalization": None,
                                          "augmentation": {"horizontal_flip": True, "vertical_flip": True}}}}
        configs_dir = output_root / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        (configs_dir / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        (destination / "preprocess_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocess_dgx.yaml")
    parser.add_argument("--output-root", help="Override the configured destination")
    parser.add_argument("--dataset", action="append", help="Process only this dataset; repeat to select several")
    parser.add_argument("--size", type=int, help="Override both dimensions, e.g. 1024")
    parser.add_argument("--resize-mode", choices=["stretch", "letterbox"])
    parser.add_argument("--split-policy", choices=["official", "scene"], default="official",
                        help="official validates supplied lists; scene explicitly discards supplied splits and uses new scene-group partitions")
    parser.add_argument("--dry-run", action="store_true", help="Read/check all selected pairs without writing")
    parser.add_argument("--limit", type=int, help="Small preparation smoke run; not a full dataset")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files; never modify sources")
    parser.add_argument("--allow-unlabeled-images", action="store_true", help="Explicitly skip image-only files and record their paths")
    parser.add_argument("--paired-only", action="store_true", help="Keep only complete image/mask pairs; report unmatched images AND masks without deleting them")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    size = [args.size, args.size] if args.size is not None else cfg.get("size", [1024, 1024])
    if len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
        parser.error("size must contain two positive integers")
    mode = args.resize_mode or cfg.get("resize_mode", "stretch")
    if mode not in {"stretch", "letterbox"}:
        parser.error("resize_mode must be stretch or letterbox")
    root = Path(args.output_root).expanduser().resolve() if args.output_root else resolve(cfg["output_root"], path.parent)
    names = args.dataset or list(cfg["datasets"])
    if any(name not in cfg["datasets"] for name in names):
        parser.error(f"Dataset must be one of {list(cfg['datasets'])}")
    for name in names:
        spec = dict(cfg["datasets"][name])
        if args.split_policy == "scene":
            if spec.get("source_manifest"):
                parser.error("--split-policy scene cannot override a source_manifest; edit its split column explicitly")
            spec.pop("split_files", None)
            print(f"{name}: explicitly using NEW scene-group splits, not source split lists", flush=True)
        report = prepare_dataset(name, spec, path.parent, root, size, mode,
                                 args.dry_run, args.limit, args.overwrite, args.allow_unlabeled_images, args.paired_only)
        print(json.dumps(report, indent=2), flush=True)
    if not args.dry_run:
        experiment = {"tasks": [str(root / "configs" / f"{name}.yaml") for name in names],
                      "data_loader": {"batch_size": 1, "num_workers": 0, "pin_memory": True, "seed": 42},
                      "model": {"kind": "quantum", "qubits": 4, "depth": 2, "grid": 4},
                      "training": {"epochs_per_task": 5, "learning_rate": .001, "seed": 42},
                      "ewc": {"strength": 100., "decay": 1., "scope": "all", "fisher_images": 16, "fisher_pixels_per_image": 4}}
        experiment_path = root / "configs" / "experiment.yaml"
        experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
        print(f"Training experiment config: {experiment_path}", flush=True)


if __name__ == "__main__":
    main()
