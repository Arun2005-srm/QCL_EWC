from PIL import Image
import torch

from continual.visualization import save_predictions
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
