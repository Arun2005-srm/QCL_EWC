import numpy as np

from .raster import read_image, read_mask


def validate_records(records, cfg, limit=None):
    """Scan raw scene values, reporting source-pixel counts before augmentation."""
    counts = np.zeros(len(cfg["classes"]), dtype=np.int64)
    ignored = 0
    items = [r for subset in records.values() for r in subset]
    selected = items if limit is None else items[:limit]
    ignore = cfg["labels"].get("ignore_index", 255)
    for record in selected:
        image = read_image(record.image_path, cfg.get("images", {}))
        mask = read_mask(record.mask_path, cfg["labels"])
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask dimensions differ: {record.sample_id}")
        valid = mask != ignore
        counts += np.bincount(mask[valid], minlength=len(counts))
        ignored += int((~valid).sum())
    return {"dataset": cfg["name"], "scenes": {k: len(v) for k, v in records.items()},
            "scanned_scenes": len(selected), "complete_scan": len(selected) == len(items),
            "class_pixels": dict(zip(cfg["classes"], counts.tolist())), "ignored_pixels": ignored}
