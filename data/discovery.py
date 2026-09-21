import csv
from pathlib import Path
import re

from .contracts import SampleRecord


def _index(section, pairing, side, recursive):
    root = Path(section["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"{side} directory not found: {root}")
    extensions = {"." + str(e).lower().lstrip(".") for e in section.get("extensions", [".png", ".jpg", ".jpeg", ".tif", ".tiff"])}
    files = root.rglob("*") if recursive else root.glob("*")
    index = {}
    suffix = pairing.get(f"{side}_suffix", "")
    for path in sorted(p for p in files if p.is_file() and p.suffix.lower() in extensions):
        stem = path.stem
        if suffix:
            if not stem.endswith(suffix):
                raise ValueError(f"Expected {side} suffix {suffix!r}: {path}")
            stem = stem[:-len(suffix)]
        identifier = (path.relative_to(root).parent / stem).as_posix() if pairing.get("method", "stem") == "relative" else stem
        if not identifier or identifier in index:
            raise ValueError(f"Duplicate or empty {side} ID {identifier!r}; use pairing.method=relative for nested folders")
        index[identifier] = path.resolve()
    return index


def discover_pairs(cfg):
    if cfg.get("manifest"):
        manifest = Path(cfg["manifest"])
        records = []
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"id", "image", "mask"}.issubset(reader.fieldnames or []):
                raise ValueError("Manifest CSV requires id,image,mask columns; group and split are optional")
            for row in reader:
                if any(not row.get(k, "").strip() for k in ("id", "image", "mask")):
                    raise ValueError("Manifest has an empty id, image, or mask")
                records.append(SampleRecord(row["id"].strip(), (manifest.parent / row["image"]).resolve(),
                                            (manifest.parent / row["mask"]).resolve(),
                                            (row.get("group") or row["id"]).strip(), (row.get("split") or "").strip()))
    else:
        pairing = cfg.get("pairing", {})
        recursive = cfg.get("recursive", True)
        images = _index(cfg["images"], pairing, "image", recursive)
        masks = _index(cfg["masks"], pairing, "mask", recursive)
        missing, orphan = sorted(images.keys() - masks.keys()), sorted(masks.keys() - images.keys())
        if missing or orphan:
            raise ValueError(f"Unpaired files: images without masks={missing[:5]}, masks without images={orphan[:5]}")
        pattern = cfg.get("grouping", {}).get("regex")
        records = []
        for identifier in sorted(images):
            group = identifier
            if pattern:
                match = re.search(pattern, identifier)
                if match is None or "group" not in match.groupdict() or not match.group("group"):
                    raise ValueError(f"grouping.regex must capture a nonempty named group 'group' for {identifier}")
                group = match.group("group")
            records.append(SampleRecord(identifier, images[identifier], masks[identifier], group))
    if not records:
        raise ValueError("No image/mask pairs found")
    for field in ("sample_id", "image_path", "mask_path"):
        values = [getattr(r, field) for r in records]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field} in dataset")
    for r in records:
        if not r.group:
            raise ValueError(f"Empty group for {r.sample_id}")
        for p in (r.image_path, r.mask_path):
            if not p.is_file():
                raise FileNotFoundError(p)
    return sorted(records, key=lambda r: r.sample_id)
