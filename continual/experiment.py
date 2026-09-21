import hashlib
from pathlib import Path

import torch

from data.config import _read, load_task_configs


def read_experiment(path):
    experiment = _read(path)
    tasks = load_task_configs(path)
    for task in tasks:
        if task["dataset"].get("transforms", {}).get("normalization"):
            raise ValueError("SAM expects RGB [0,1]; set dataset.transforms.normalization: null")
    return experiment, tasks


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
