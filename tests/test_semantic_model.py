import json
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from models.sam_segmenter import UniversalSAM
from continual.metrics import segmentation_loss
from data import load_config, build_records
from smoke import make_fixture
from prepare_native_tiles import prepare


class FakeEncoder(nn.Module):
    img_size = 32

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 1)

    def forward(self, x):
        return self.conv(F.adaptive_avg_pool2d(x, 4))


class FakeSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = FakeEncoder()

    def preprocess(self, x):
        return F.pad(x / 255, (0, 32 - x.shape[-1], 0, 32 - x.shape[-2]))


@pytest.fixture(autouse=True)
def small_cpu_workload():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("kind", ["classical", "quantum"])
def test_semantic_gradients_frozen_encoder_and_roundtrip(kind):
    if kind == "quantum":
        pytest.importorskip("pennylane")
    torch.manual_seed(5)
    model = UniversalSAM(5, sam=FakeSAM(), kind=kind, decoder="semantic",
                         decoder_width=16, grid=2, gated=True, encoder_batch_size=1)
    image = torch.rand(2, 3, 24, 32)
    target = torch.randint(0, 5, (2, 24, 32))
    target[:, :2] = 255
    model.train()
    logits = model(image)
    assert logits.shape == (2, 5, 24, 32)
    segmentation_loss(logits, target).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
    assert not model.sam.training
    assert not hasattr(model, "class_prompts")
    saved = model.trainable_state()
    with torch.no_grad():
        model.semantic_decoder.head.weight.add_(1)
    model.load_trainable_state(saved)
    assert torch.allclose(model(image), logits)


def test_semantic_can_overfit_a_tiny_pattern():
    torch.manual_seed(4)
    model = UniversalSAM(2, sam=FakeSAM(), kind="classical", decoder="semantic",
                         decoder_width=16, grid=2, gated=True)
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, :, 16:] = 1
    target = image[:, 0].long()
    opt = torch.optim.Adam(model.semantic_decoder.parameters(), lr=.003)
    first = segmentation_loss(model(image), target).item()
    for _ in range(40):
        opt.zero_grad()
        loss = segmentation_loss(model(image), target)
        loss.backward()
        opt.step()
    logits = model(image)
    assert segmentation_loss(logits, target).item() < first * .3
    assert (logits.argmax(1) == target).float().mean() > .98


def test_native_tiles_preserve_pixels_and_scene_splits(tmp_path):
    path = make_fixture(tmp_path / "original")
    cfg = load_config(path)
    records = build_records(cfg["dataset"])
    sources = tmp_path / "sources.jsonl"
    sources.write_text("\n".join(json.dumps(dict(id=r.sample_id, source_id=r.sample_id,
                                               source_image=str(r.image_path), source_mask=str(r.mask_path)))
                                 for subset in records.values() for r in subset), encoding="utf-8")
    out = tmp_path / "tiles"
    report = prepare(path, sources, out, size=32)
    generated = load_config(out / "dataset.yaml")
    tiles = build_records(generated["dataset"])
    assert all(report["tile_counts"][s] >= len(records[s]) for s in records)
    for split in records:
        assert {r.group for r in tiles[split]} == {r.group for r in records[split]}
    info = json.loads((out / "tiles.jsonl").read_text().splitlines()[0])
    original = np.asarray(Image.open(info["source_image"]))
    output = np.asarray(Image.open(out / "images" / (info["id"] + ".png")))
    assert np.array_equal(output, original[:32, :32])
    assert report["cross_dataset_overlap_audited"] is False
    with pytest.raises(FileExistsError):
        prepare(path, sources, out, size=32)
