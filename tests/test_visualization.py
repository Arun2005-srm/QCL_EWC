from PIL import Image
import torch

from continual.visualization import save_predictions, save_smoke_comparison
from data import load_config, build_records
from data.dataset import SegmentationDataset
from smoke import make_fixture


def test_prediction_panels(tmp_path):
    cfg = load_config(make_fixture(tmp_path / "fixture"))["dataset"]
    ds = SegmentationDataset(build_records(cfg)["test"], cfg)
    model = torch.nn.Conv2d(3, 5, 1)
    save_predictions(model, ds, "cpu", tmp_path / "panels", limit=1)
    with Image.open(tmp_path / "panels" / "sample_000.png") as panel:
        assert panel.size == (96, 32)


def test_smoke_comparison_restores_model(tmp_path):
    class Model(torch.nn.Conv2d):
        def trainable_state(self):
            return {k: v.detach().clone() for k, v in self.state_dict().items()}

        def load_trainable_state(self, state):
            self.load_state_dict(state)

    cfg = load_config(make_fixture(tmp_path / "fixture"))["dataset"]
    ds = SegmentationDataset(build_records(cfg)["test"], cfg)
    model = Model(3, 5, 1)
    original = model.trainable_state()
    altered = {k: v + .2 for k, v in original.items()}
    destination = tmp_path / "comparison.png"
    save_smoke_comparison(model, ds, "cpu", [altered, original],
                          [[{"miou": .1}], [{"miou": .2}, {"miou": .3}]], destination)
    assert model.training
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    with Image.open(destination) as panel:
        assert panel.size == (1280, 660)
