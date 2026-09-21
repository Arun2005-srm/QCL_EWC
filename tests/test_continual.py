from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from continual.ewc import OnlineEWC
from continual.metrics import segmentation_loss, Confusion
from data import load_config
from smoke import make_fixture
from train import run_training


class TinySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 1)

    def forward(self, image):
        return self.conv(image)

    def trainable_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_trainable_state(self, state):
        self.load_state_dict(state)


def test_fisher_penalty_and_roundtrip():
    torch.manual_seed(12)
    model = TinySegmenter()
    loader = DataLoader([{"image": torch.rand(3, 4, 4), "mask": torch.randint(0, 5, (4, 4))}])
    ewc = OnlineEWC(strength=10)
    ewc.consolidate(model, loader, "cpu", max_images=1, pixels_per_image=2)
    assert all((f >= 0).all() for f in ewc.fisher.values())
    assert sum(f.sum() for f in ewc.fisher.values()) > 0
    assert ewc.penalty(model).item() == 0
    with torch.no_grad():
        model.conv.weight.add_(.1)
    assert ewc.penalty(model).item() > 0
    copy = OnlineEWC()
    copy.load_state_dict(ewc.state_dict())
    assert torch.equal(copy.penalty(model), ewc.penalty(model))


def test_fisher_squares_individual_gradients():
    model = nn.Conv2d(1, 2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    # Opposite targets have cancelling mean gradients, but nonzero Fisher.
    loader = DataLoader([{"image": torch.ones(1, 1, 2), "mask": torch.tensor([[0, 1]])}])
    ewc = OnlineEWC()
    ewc.consolidate(model, loader, "cpu", max_images=1, pixels_per_image=2)
    assert torch.allclose(ewc.fisher["weight"], torch.full_like(model.weight, .25))


def test_ignore_loss_and_metrics():
    logits = torch.randn(1, 5, 2, 2, requires_grad=True)
    masks = torch.full((1, 2, 2), 255)
    loss = segmentation_loss(logits, masks)
    loss.backward()
    assert loss.item() == 0 and logits.grad.abs().sum() == 0
    meter = Confusion(5)
    meter.update(logits, masks)
    assert meter.result()["miou"] is None


def test_two_task_training_and_resume(tmp_path):
    tasks = [load_config(make_fixture(tmp_path / name)) for name in ("one", "two")]
    for i, cfg in enumerate(tasks):
        cfg["dataset"]["name"] = str(i)
        cfg["dataset"]["tiling"]["enabled"] = False
    experiment = {"training": {"epochs_per_task": 1, "seed": 42},
                  "ewc": {"strength": 1., "fisher_images": 1, "fisher_pixels_per_image": 2}}
    torch.manual_seed(42)
    original = TinySegmenter()
    result = run_training(experiment, tasks, original, tmp_path / "run", torch.device("cpu"), "synthetic")
    assert [len(row) for row in result["matrix"]] == [1, 2]
    assert len(result["forgetting_miou"]) == 1
    resumed = TinySegmenter()
    again = run_training(experiment, tasks, resumed, tmp_path / "resume", torch.device("cpu"), "synthetic", tmp_path / "run" / "task_00.pt")
    assert again["matrix"] == result["matrix"]
    for name, value in original.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value)
