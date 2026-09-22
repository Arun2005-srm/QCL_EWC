import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import tifffile
import yaml

from prepare_datasets import discover, prepare_dataset, resize_pair, apply_split_files
from data import load_config, build_dataloaders


def make_source(root, nested=False):
    for i in range(3):
        folder = root / f"region_{i}" if nested else root
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "labels").mkdir(parents=True, exist_ok=True)
        stem = "scene" if nested else f"scene_{i}"
        mask = np.tile(np.array([0, 4, 8, 4], dtype=np.uint8), (6, 1))
        image = np.stack([mask * 20, mask * 20, mask * 20], axis=-1)
        tifffile.imwrite(folder / "images" / f"{stem}.tif", image, photometric="rgb")
        tifffile.imwrite(folder / "labels" / f"{stem}.tif", mask)
    return {"source": str(root), "image_dirs": ["images"], "mask_dirs": ["labels"],
            "label_mapping": {0: 255, 4: 4, 8: 1, 255: 255}}


def test_nested_resize_and_project_integration(tmp_path):
    spec = make_source(tmp_path / "source", nested=True)
    source_bytes = {p: p.read_bytes() for p in (tmp_path / "source").rglob("*.tif")}
    out = tmp_path / "prepared"
    report = prepare_dataset("openearthmap", spec, tmp_path, out, [1024, 1024], "stretch")
    assert report["processed_pairs"] == 3
    images = list((out / "openearthmap" / "images").glob("*.png"))
    assert len(images) == 3 and len({p.name for p in images}) == 3
    for p in (out / "openearthmap" / "masks").glob("*.png"):
        with Image.open(p) as mask:
            assert mask.size == (1024, 1024) and mask.mode == "L"
            assert set(np.unique(mask).tolist()) == {0, 4, 8}
    cfg = load_config(out / "configs" / "openearthmap.yaml")
    assert not cfg["dataset"]["tiling"]["enabled"]
    loaders = build_dataloaders(cfg["dataset"], cfg["data_loader"])
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape == (1, 3, 1024, 1024)
    assert batch["mask"].shape == (1, 1024, 1024)
    assert set(batch["mask"].unique().tolist()) == {255, 4, 1}
    assert all(p.read_bytes() == content for p, content in source_bytes.items())


def test_letterbox_preserves_ids_and_marks_padding():
    image = np.full((4, 8, 3), 120, dtype=np.uint8)
    mask = np.full((4, 8), 3, dtype=np.uint8)
    rgb, labels, size = resize_pair(image, mask, [16, 16], "letterbox")
    assert size == [8, 16] and rgb.size == (16, 16)
    assert set(np.unique(labels).tolist()) == {3, 255}
    assert (np.asarray(labels)[8:] == 255).all()


def test_dry_run_writes_nothing(tmp_path):
    spec = make_source(tmp_path / "source")
    out = tmp_path / "prepared"
    report = prepare_dataset("test", spec, tmp_path, out, [1024, 1024], "stretch", dry_run=True)
    assert report["processed_pairs"] == 3 and not out.exists()


def test_unpaired_images_require_explicit_skip(tmp_path):
    spec = make_source(tmp_path / "source")
    Image.new("RGB", (4, 6)).save(tmp_path / "source" / "images" / "unlabeled.png")
    with pytest.raises(ValueError, match="Unpaired"):
        discover(spec, tmp_path)
    pairs, skipped = discover(spec, tmp_path, allow_unlabeled=True)
    assert len(pairs) == 3 and len(skipped) == 1


def test_existing_destination_not_overwritten(tmp_path):
    spec = make_source(tmp_path / "source")
    out = tmp_path / "prepared"
    prepare_dataset("test", spec, tmp_path, out, [16, 16], "stretch")
    with pytest.raises(FileExistsError):
        prepare_dataset("test", spec, tmp_path, out, [16, 16], "stretch")


def test_source_destination_separation(tmp_path):
    spec = make_source(tmp_path / "source")
    with pytest.raises(ValueError, match="separate"):
        prepare_dataset("test", spec, tmp_path, tmp_path / "source", [16, 16], "stretch")


def test_official_manifest_preserved(tmp_path):
    spec = make_source(tmp_path / "source")
    pairs, _ = discover(spec, tmp_path)
    path = tmp_path / "official.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image", "mask", "group", "split"])
        for p, split in zip(pairs, ("train", "val", "test")):
            writer.writerow([p.identifier, p.image, p.mask, p.group, split])
    spec["source_manifest"] = str(path)
    out = tmp_path / "prepared"
    prepare_dataset("test", spec, tmp_path, out, [16, 16], "stretch")
    cfg = load_config(out / "configs" / "test.yaml")
    assert cfg["dataset"]["split"]["strategy"] == "manifest"
    loaders = build_dataloaders(cfg["dataset"], cfg["data_loader"])
    assert all(len(loader.dataset) == 1 for loader in loaders.values())


def test_unexpected_labels_rejected(tmp_path):
    spec = make_source(tmp_path / "source")
    spec["label_mapping"].pop(8)
    with pytest.raises(ValueError, match="Unexpected raw"):
        prepare_dataset("test", spec, tmp_path, tmp_path / "out", [16, 16], "stretch", dry_run=True)


def test_cli_generates_experiment(tmp_path):
    spec = make_source(tmp_path / "source")
    config = tmp_path / "prepare.yaml"
    config.write_text(yaml.safe_dump({"output_root": str(tmp_path / "out"), "datasets": {"test": spec}}))
    script = Path(__file__).resolve().parents[1] / "prepare_datasets.py"
    result = subprocess.run([sys.executable, str(script), "--config", str(config), "--limit", "1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "out" / "test" / "preprocess_report.json").read_text())
    assert report["limited_run"] and report["size"] == [1024, 1024]
    assert (tmp_path / "out" / "configs" / "experiment.yaml").is_file()


def test_split_text_lists_support_stems_and_paths(tmp_path):
    root = tmp_path / "source"
    spec = make_source(root, nested=True)
    spec["split_files"] = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
    for i, split in enumerate(("train", "val", "test")):
        (root / f"{split}.txt").write_text(f"region_{i}/images/scene.tif\n")
    pairs, skipped = discover(spec, tmp_path)
    pairs = apply_split_files(pairs, spec, tmp_path, skipped)
    assert [p.split for p in pairs] == ["train", "val", "test"]


def test_tile_lists_rejected_for_whole_scenes(tmp_path):
    root = tmp_path / "source"
    spec = make_source(root)
    spec["split_files"] = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
    (root / "train.txt").write_text("scene_0_17\n")
    pairs, _ = discover(spec, tmp_path)
    with pytest.raises(ValueError, match="TILE"):
        apply_split_files(pairs, spec, tmp_path)


def test_split_lists_cannot_leak_same_scene(tmp_path):
    root = tmp_path / "source"
    spec = make_source(root)
    spec["split_files"] = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
    for split in ("train", "val", "test"):
        (root / f"{split}.txt").write_text("scene_0.tif\n")
    pairs, _ = discover(spec, tmp_path)
    with pytest.raises(ValueError, match="multiple splits"):
        apply_split_files(pairs, spec, tmp_path)


def test_paired_only_reports_both_kinds_and_keeps_sources(tmp_path):
    root = tmp_path / "source"
    spec = make_source(root)
    orphan = root / "labels" / "orphan.tif"
    unlabeled = root / "images" / "unlabeled.tif"
    tifffile.imwrite(orphan, np.zeros((6, 4), dtype=np.uint8))
    tifffile.imwrite(unlabeled, np.zeros((6, 4, 3), dtype=np.uint8), photometric="rgb")
    spec["split_files"] = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
    for i, split in enumerate(("train", "val", "test")):
        (root / f"{split}.txt").write_text(f"scene_{i}.tif\n" + ("orphan.tif\nunlabeled.tif\n" if i == 0 else ""))
    with pytest.raises(ValueError, match="Unpaired"):
        discover(spec, tmp_path, allow_unlabeled=True)
    report = prepare_dataset("test", spec, tmp_path, tmp_path / "out", [16, 16], "stretch", paired_only=True)
    assert report["processed_pairs"] == 3
    assert report["skipped_orphan_masks"] == [str(orphan.resolve())]
    assert report["skipped_unlabeled_images"] == [str(unlabeled.resolve())]
    assert report["source_split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert orphan.is_file() and unlabeled.is_file()
