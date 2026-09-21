"""YAML inheritance and paths resolved relative to the file defining them."""
from copy import deepcopy
import math
import os
from pathlib import Path
import re

import yaml


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        # A new label map must never silently retain IDs from another dataset.
        if isinstance(value, dict) and isinstance(result.get(key), dict) and key not in {"mapping", "colors"}:
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _path(value, parent):
    value = os.path.expandvars(str(value))
    if re.search(r"\$\{[^}]+\}|\$[A-Za-z_]\w*|%[A-Za-z_]\w*%", value):
        raise ValueError(f"Unset environment variable in path: {value}")
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else parent / path).resolve())


def _read(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Circular defaults inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    defaults = cfg.pop("defaults", None)
    base = _read(_path(defaults, path.parent), (*stack, path)) if defaults else {}
    dataset = cfg.get("dataset", {})
    for name in ("images", "masks"):
        if "root" in dataset.get(name, {}):
            dataset[name]["root"] = _path(dataset[name]["root"], path.parent)
    if "manifest" in dataset:
        dataset["manifest"] = _path(dataset["manifest"], path.parent)
    if "tasks" in cfg:
        cfg["tasks"] = [_path(item, path.parent) for item in cfg["tasks"]]
    return _merge(base, cfg)


def _size(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(x) is not int or x <= 0 for x in value):
        raise ValueError(f"{name} must contain two positive integers [height, width]")


def validate_config(cfg):
    d = cfg.get("dataset")
    if not isinstance(d, dict) or not d.get("name"):
        raise ValueError("dataset.name is required")
    if not d.get("manifest") and not all(d.get(k, {}).get("root") for k in ("images", "masks")):
        raise ValueError("Provide dataset.manifest or both images.root and masks.root")
    classes = d.get("classes", [])
    if not classes or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
        raise ValueError("dataset.classes must be a nonempty list of unique names")
    labels = d.get("labels", {})
    ignore = labels.get("ignore_index", 255)
    if type(ignore) is not int or 0 <= ignore < len(classes):
        raise ValueError("ignore_index must be an integer outside the class indices")
    encoding = labels.get("encoding", "index")
    if encoding not in {"index", "rgb", "binary"}:
        raise ValueError("labels.encoding must be index, rgb, or binary")
    mapping = labels.get("colors" if encoding == "rgb" else "mapping", {})
    if encoding != "binary" and not mapping:
        raise ValueError("An explicit labels.mapping or labels.colors is required")
    if any(type(v) is not int or (v != ignore and not 0 <= v < len(classes)) for v in mapping.values()):
        raise ValueError("Label targets must be class indices or ignore_index")
    if encoding == "index" and any(type(k) is not int for k in mapping):
        raise ValueError("Index mapping keys must be integers")
    if encoding == "rgb":
        for color in mapping:
            try:
                parts = [int(v.strip()) for v in str(color).split(",")]
            except ValueError as exc:
                raise ValueError(f"Invalid RGB color: {color}") from exc
            if len(parts) != 3 or any(v < 0 or v > 255 for v in parts):
                raise ValueError(f"Invalid RGB color: {color}; use 'R,G,B'")
    if encoding == "binary" and len(classes) != 2:
        raise ValueError("Binary encoding requires exactly two classes")
    pairing = d.get("pairing", {})
    if pairing.get("method", "stem") not in {"stem", "relative"}:
        raise ValueError("pairing.method must be stem or relative")
    split = d.get("split", {})
    strategy = split.get("strategy", "group")
    if strategy not in {"group", "random", "manifest"}:
        raise ValueError("split.strategy must be group, random, or manifest")
    if strategy == "manifest" and not d.get("manifest"):
        raise ValueError("split.strategy=manifest requires dataset.manifest")
    ratios = split.get("ratios", {"train": .7, "val": .15, "test": .15})
    if set(ratios) != {"train", "val", "test"} or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in ratios.values()) or not math.isclose(sum(ratios.values()), 1):
        raise ValueError("split.ratios must specify train, val, test and sum to 1")
    transform = d.get("transforms", {})
    _size(transform.get("image_size", [512, 512]), "transforms.image_size")
    tile = d.get("tiling", {})
    if tile.get("enabled", False):
        size = tile.get("size", [512, 512])
        stride = tile.get("stride", size)
        _size(size, "tiling.size")
        _size(stride, "tiling.stride")
        if any(s > n for s, n in zip(stride, size)):
            raise ValueError("tiling.stride cannot exceed size (would leave uncovered pixels)")
    image = d.get("images", {})
    scale = image.get("scale", 255.0)
    if not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("images.scale must be finite and positive")
    bands = image.get("bands")
    if bands is not None and (len(bands) != 3 or any(type(b) is not int or b < 0 for b in bands)):
        raise ValueError("images.bands must specify three zero-based channel indices")
    normalization = transform.get("normalization")
    if normalization:
        mean, std = normalization.get("mean", []), normalization.get("std", [])
        if len(mean) != 3 or len(std) != 3 or any(not math.isfinite(v) for v in [*mean, *std]) or any(v <= 0 for v in std):
            raise ValueError("normalization requires three finite means and positive std values")
    loader = cfg.get("data_loader", {})
    for name, minimum, default in (("batch_size", 1, 1), ("num_workers", 0, 0)):
        if type(loader.get(name, default)) is not int or loader.get(name, default) < minimum:
            raise ValueError(f"data_loader.{name} must be an integer >= {minimum}")
    return cfg


def load_config(path):
    """Load and validate a single dataset YAML, including inherited defaults."""
    return validate_config(_read(path))


def load_task_configs(path):
    """Return ordered, independent dataset configs sharing one class vocabulary."""
    experiment = _read(path)
    tasks = experiment.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Experiment must contain a nonempty tasks list of dataset YAML paths")
    configs = []
    for task in tasks:
        cfg = load_config(task)
        cfg["data_loader"] = _merge(cfg.get("data_loader", {}), experiment.get("data_loader", {}))
        configs.append(validate_config(cfg))
    signature = lambda c: (c["dataset"]["classes"], c["dataset"].get("labels", {}).get("ignore_index", 255))
    if any(signature(c) != signature(configs[0]) for c in configs[1:]):
        raise ValueError("Continual tasks must share ordered classes and ignore_index")
    names = [c["dataset"]["name"] for c in configs]
    if len(names) != len(set(names)):
        raise ValueError("Continual task dataset names must be unique")
    return configs
