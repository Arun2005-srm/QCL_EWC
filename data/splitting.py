from dataclasses import replace
import random

SPLITS = ("train", "val", "test")


def split_records(records, cfg):
    strategy = cfg.get("strategy", "group")
    result = {k: [] for k in SPLITS}
    if strategy == "manifest":
        for record in records:
            if record.split not in SPLITS:
                raise ValueError(f"Manifest split must be train, val or test: {record.sample_id}")
            result[record.split].append(record)
    else:
        if any(r.split for r in records):
            raise ValueError("Manifest already specifies splits; use split.strategy=manifest to preserve them")
        ratios = cfg.get("ratios", {"train": .7, "val": .15, "test": .15})
        key = (lambda r: r.group) if strategy == "group" else (lambda r: r.sample_id)
        units = sorted({key(r) for r in records})
        random.Random(cfg.get("seed", 42)).shuffle(units)
        active = [s for s in SPLITS if ratios[s] > 0]
        if len(units) < len(active):
            raise ValueError("Not enough independent scenes/groups for requested nonempty splits")
        # Closest integer allocation, with at least one unit per requested split.
        counts = {s: (max(1, int(len(units) * ratios[s])) if s in active else 0) for s in SPLITS}
        while sum(counts.values()) > len(units):
            candidates = [s for s in active if counts[s] > 1]
            s = max(candidates, key=lambda s: counts[s] - len(units) * ratios[s])
            counts[s] -= 1
        while sum(counts.values()) < len(units):
            s = max(active, key=lambda s: len(units) * ratios[s] - counts[s])
            counts[s] += 1
        assignment, offset = {}, 0
        for s in SPLITS:
            assignment.update({unit: s for unit in units[offset:offset + counts[s]]})
            offset += counts[s]
        for record in records:
            s = assignment[key(record)]
            result[s].append(replace(record, split=s))
    group_splits = {}
    for s, subset in result.items():
        for record in subset:
            if record.group in group_splits and group_splits[record.group] != s:
                raise ValueError(f"Scene/group leakage across splits: {record.group}")
            group_splits[record.group] = s
    if not result["train"]:
        raise ValueError("The training split is empty")
    return result
