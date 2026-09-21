from copy import deepcopy
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tifffile
import torch
import yaml

from data import load_config, load_task_configs, build_records, build_dataloaders, save_manifest
from data.config import validate_config
from data.dataset import SegmentationDataset
from data.raster import read_image, read_mask
from data.validation import validate_records
from smoke import make_fixture


@pytest.fixture
def cfg(tmp_path):
    return load_config(make_fixture(tmp_path))


def test_contract_padding_and_scene_isolation(cfg):
    records = build_records(cfg["dataset"])
    loaders = build_dataloaders(cfg["dataset"], cfg["data_loader"], records=records)
    groups = [{r.group for r in records[s]} for s in ("train", "val", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["mask"].shape == (2, 32, 32)
    assert batch["image"].dtype == torch.float32 and batch["mask"].dtype == torch.int64
    assert 0 <= batch["image"].min() <= batch["image"].max() <= 1
    edge = loaders["val"].dataset[3]
    assert edge["valid_size"].tolist() == [8, 16]
    assert (edge["mask"][8:] == 255).all() and (edge["mask"][:, 16:] == 255).all()


def test_manifest_roundtrip(cfg, tmp_path):
    before = build_records(cfg["dataset"])
    path = tmp_path / "splits.csv"
    save_manifest(before, path)
    cfg["dataset"].update(manifest=str(path), split={"strategy": "manifest"})
    assert build_records(cfg["dataset"]) == before


def test_splits_reproducible(cfg):
    assert build_records(cfg["dataset"]) == build_records(cfg["dataset"])


def test_group_leakage_rejected(cfg, tmp_path):
    records = build_records(cfg["dataset"])
    path = tmp_path / "splits.csv"
    save_manifest(records, path)
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["group"] = "same_scene"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg["dataset"].update(manifest=str(path), split={"strategy": "manifest"})
    with pytest.raises(ValueError, match="leakage"):
        build_records(cfg["dataset"])


def test_unknown_label_fails(cfg):
    cfg["dataset"]["labels"]["mapping"].pop(4)
    with pytest.raises(ValueError, match="Unmapped"):
        validate_records(build_records(cfg["dataset"]), cfg["dataset"])


def test_unpaired_file_rejected(cfg):
    Image.new("L", (5, 5)).save(Path(cfg["dataset"]["masks"]["root"]) / "orphan_mask.png")
    with pytest.raises(ValueError, match="Unpaired"):
        build_records(cfg["dataset"])


def test_duplicate_id_rejected(cfg):
    Image.new("RGB", (5, 5)).save(Path(cfg["dataset"]["images"]["root"]) / "scene_0.jpg")
    with pytest.raises(ValueError, match="Duplicate"):
        build_records(cfg["dataset"])


def test_rgb_masks(tmp_path):
    path = tmp_path / "rgb.png"
    Image.fromarray(np.array([[[255, 0, 0], [0, 0, 0]]], dtype=np.uint8)).save(path)
    result = read_mask(path, {"encoding": "rgb", "colors": {"255,0,0": 1, "0,0,0": 255}})
    assert result.tolist() == [[1, 255]]


def test_palette_ids_preserved(tmp_path):
    path = tmp_path / "palette.png"
    image = Image.fromarray(np.array([[0, 1]], dtype=np.uint8)).convert("P")
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    image.save(path)
    assert read_mask(path, {"encoding": "index", "mapping": {0: 0, 1: 4}}).tolist() == [[0, 4]]


def test_binary_mask(tmp_path):
    path = tmp_path / "binary.png"
    Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)).save(path)
    assert read_mask(path, {"encoding": "binary", "threshold": 127, "ignore_values": [128]}).tolist() == [[0, 255, 1]]


def test_multiband_tiff_requires_explicit_bands(tmp_path):
    path = tmp_path / "four_band.tif"
    tifffile.imwrite(path, np.full((8, 8, 4), 1024, dtype=np.uint16), photometric="rgb")
    with pytest.raises(ValueError, match="three RGB"):
        read_image(path, {"scale": 65535})
    assert read_image(path, {"scale": 65535, "bands": [2, 1, 0]}).shape == (8, 8, 3)


def test_tiff_index_mask(tmp_path):
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, np.array([[0, 512]], dtype=np.uint16))
    assert read_mask(path, {"mapping": {0: 0, 512: 1}}).tolist() == [[0, 1]]


def test_resize_and_flips_keep_alignment(cfg):
    d = cfg["dataset"]
    d["tiling"]["enabled"] = False
    d["transforms"].update(image_size=[40, 48], augmentation={"horizontal_flip": True, "vertical_flip": True})
    ds = SegmentationDataset(build_records(d)["train"], d, training=True)
    torch.manual_seed(3)
    sample = ds[0]
    assert torch.equal((sample["image"][0] * 255 / 40).round().long(), sample["mask"])


def test_dimension_mismatch(cfg):
    Image.new("L", (5, 5)).save(Path(cfg["dataset"]["masks"]["root"]) / "scene_0_mask.png")
    with pytest.raises(ValueError, match="dimensions"):
        build_dataloaders(cfg["dataset"])


def test_defaults_relative_paths_and_cycle(tmp_path):
    base = make_fixture(tmp_path)
    child = tmp_path / "nested" / "child.yaml"
    child.parent.mkdir()
    child.write_text("defaults: ../dataset.yaml\ndataset:\n  name: child\n")
    assert load_config(child)["dataset"]["images"]["root"] == str(tmp_path / "images")
    base.write_text("defaults: nested/child.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(child)


def test_task_vocabulary_mismatch(tmp_path):
    first = make_fixture(tmp_path / "first")
    second = make_fixture(tmp_path / "second")
    content = yaml.safe_load(second.read_text())
    content["dataset"]["name"] = "second"
    content["dataset"]["classes"][0] = "different"
    second.write_text(yaml.safe_dump(content))
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump({"tasks": [str(first), str(second)]}))
    with pytest.raises(ValueError, match="ordered classes"):
        load_task_configs(experiment)


@pytest.mark.parametrize("change", [
    {"split": {"ratios": {"train": 1, "val": 1, "test": 0}}},
    {"labels": {"mapping": {0: 20}}},
    {"transforms": {"image_size": [0, 32]}},
    {"tiling": {"enabled": True, "size": [10, 10], "stride": [20, 20]}},
])
def test_invalid_config(cfg, change):
    cfg["dataset"].update(change)
    with pytest.raises(ValueError):
        validate_config(cfg)
