from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from continual.ewc import OnlineEWC
from continual.metrics import segmentation_loss, Confusion, inner_boundary, evaluate
from data import load_config
from smoke import make_fixture
from train import run_training, validate_continuation


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
    assert meter.result()["pixel_accuracy"] is None
    assert meter.result()["mean_boundary_iou"] is None


def test_confusion_accuracy_dice_iou_and_boundary():
    target = torch.tensor([[[0, 0], [1, 255]]])
    pred = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.nn.functional.one_hot(pred, 3).permute(0, 3, 1, 2).float()
    meter = Confusion(3)
    meter.update(logits, target)
    result = meter.result()
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)
    assert result["iou"] == [.5, .5, None]
    assert result["dice"] == pytest.approx(2 / 3)
    # Every pixel is within one pixel of void; none is boundary-evaluable.
    assert result["mean_boundary_iou"] is None


def test_boundary_matches_repeated_square_erosion():
    torch.manual_seed(7)
    mask = torch.rand(2, 1, 17, 19) > .1
    mask[0] = True  # Includes image-truncated objects.
    eroded = mask.clone()
    for _ in range(3):
        padded = torch.nn.functional.pad(eroded, (1, 1, 1, 1), value=False)
        eroded = torch.stack([padded[..., y:y + 17, x:x + 19]
                             for y in range(3) for x in range(3)]).all(0)
    assert torch.equal(inner_boundary(mask, 3), mask & ~eroded)
    target = mask[:, 0].long()
    logits = torch.nn.functional.one_hot(target, 2).permute(0, 3, 1, 2).float()
    meter = Confusion(2)
    meter.update(logits, target)
    assert meter.result()["mean_boundary_iou"] == 1.


def test_evaluate_reports_loss_without_gradients():
    model = TinySegmenter()
    batch = {"image": torch.rand(1, 3, 5, 5), "mask": torch.randint(0, 5, (1, 5, 5))}
    expected = segmentation_loss(model(batch["image"]), batch["mask"]).item()
    result = evaluate(model, [batch], "cpu", 5)
    assert result["loss"] == pytest.approx(expected)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("settings", [{}, {"optimizer": "adamw", "schedule": "cosine", "weight_decay": .0001}])
def test_two_task_training_and_resume(tmp_path, settings):
    tasks = [load_config(make_fixture(tmp_path / name)) for name in ("one", "two")]
    for i, cfg in enumerate(tasks):
        cfg["dataset"]["name"] = str(i)
        cfg["dataset"]["tiling"]["enabled"] = False
    experiment = {"training": {"epochs_per_task": 1, "seed": 42, **settings},
                  "ewc": {"strength": 1., "fisher_images": 1, "fisher_pixels_per_image": 2}}
    torch.manual_seed(42)
    original = TinySegmenter()
    result = run_training(experiment, tasks, original, tmp_path / "run", torch.device("cpu"), "synthetic")
    assert [len(row) for row in result["matrix"]] == [1, 2]
    assert len(result["forgetting_miou"]) == 1
    assert result["history"][0]["training"]["loss"] > 0
    assert result["history"][0]["validation"]["loss"] > 0
    assert (tmp_path / "run" / "history.json").exists()
    resumed = TinySegmenter()
    again = run_training(experiment, tasks, resumed, tmp_path / "resume", torch.device("cpu"), "synthetic", tmp_path / "run" / "task_00.pt")
    assert again["matrix"] == result["matrix"]
    for name, value in original.state_dict().items():
        assert torch.equal(resumed.state_dict()[name], value)


def test_append_task_matches_uninterrupted_run(tmp_path):
    tasks = [load_config(make_fixture(tmp_path / name)) for name in ("first", "second")]
    for i, cfg in enumerate(tasks):
        cfg["dataset"]["name"] = str(i)
        cfg["dataset"]["tiling"]["enabled"] = False
    experiment = {"training": {"epochs_per_task": 1, "seed": 42},
                  "ewc": {"strength": 3., "fisher_images": 1, "fisher_pixels_per_image": 2}}
    torch.manual_seed(42)
    whole = TinySegmenter()
    full = run_training(experiment, tasks, whole, tmp_path / "full", torch.device("cpu"), "sam")
    torch.manual_seed(42)
    first = TinySegmenter()
    run_training(experiment, tasks[:1], first, tmp_path / "first_run", torch.device("cpu"), "sam")
    previous = tmp_path / "first_run" / "task_00.pt"
    saved = torch.load(previous, weights_only=True)
    assert saved["ewc"]["anchor"] and saved["ewc"]["fisher"]
    appended = TinySegmenter()
    extended = run_training(experiment, tasks, appended, tmp_path / "extended", torch.device("cpu"), "sam", previous, extend=True)
    assert extended["matrix"] == full["matrix"]
    assert len(extended["history"]) == 2
    assert all(torch.equal(p, appended.state_dict()[n]) for n, p in whole.state_dict().items())


def test_continuation_rejects_changed_protocol_and_incomplete_task():
    old = {"tasks": [{"dataset": "original"}], "experiment": {"training": {"seed": 42}}, "sam_sha256": "sam"}
    saved = {"specs": old, "task": 1, "epoch": 0}
    new = deepcopy(old)
    new["tasks"].append({"dataset": "next"})
    validate_continuation(saved, new)
    changed = deepcopy(new)
    changed["experiment"]["training"]["seed"] = 43
    with pytest.raises(ValueError, match="settings"):
        validate_continuation(saved, changed)
    saved["epoch"] = 1
    with pytest.raises(ValueError, match="completed"):
        validate_continuation(saved, new)


def test_two_workers_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import train
    task = load_config(make_fixture(tmp_path / "data"))
    task["dataset"]["tiling"]["enabled"] = False
    task["dataset"]["transforms"]["augmentation"] = {
        "horizontal_flip": True, "vertical_flip": True}
    task["data_loader"].update(num_workers=2, persistent_workers=False, batch_size=2)
    experiment = {"training": {"epochs_per_task": 2, "seed": 42},
                  "ewc": {"strength": 1., "fisher_images": 1, "fisher_pixels_per_image": 1}}
    boundary = tmp_path / "epoch_one.pt"
    original_save = train.atomic_save

    def save_boundary(state, path):
        original_save(state, path)
        if state["task"] == 0 and state["epoch"] == 1:
            original_save(state, boundary)

    monkeypatch.setattr(train, "atomic_save", save_boundary)
    torch.manual_seed(42)
    model = TinySegmenter()
    full = run_training(experiment, [task], model, tmp_path / "full", torch.device("cpu"), "sam")
    resumed = TinySegmenter()
    again = run_training(experiment, [task], resumed, tmp_path / "resumed", torch.device("cpu"), "sam", boundary)
    assert again["history"] == full["history"]
    assert again["matrix"] == full["matrix"]
    assert all(torch.equal(p, resumed.state_dict()[n]) for n, p in model.state_dict().items())


def test_training_rejects_persistent_workers(tmp_path):
    task = {"data_loader": {"num_workers": 2, "persistent_workers": True}}
    with pytest.raises(ValueError, match="persistent_workers=false"):
        run_training({}, [task], TinySegmenter(), tmp_path, torch.device("cpu"), "sam")
