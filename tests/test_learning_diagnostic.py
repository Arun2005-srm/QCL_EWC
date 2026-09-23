import json
import torch
from data import load_config, build_records
from data.dataset import SegmentationDataset
from smoke import make_fixture
from verify_learning import diagnostic


class TinyModel(torch.nn.Conv2d):
    def __init__(self):
        super().__init__(3, 5, 1)

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters()}


def test_diagnostic_separates_train_val_and_never_claims_benchmark(tmp_path):
    cfg = load_config(make_fixture(tmp_path / "fixture"))["dataset"]
    cfg["tiling"]["enabled"] = False
    records = build_records(cfg)
    train = SegmentationDataset(records["train"][:2], cfg, training=False)
    val = SegmentationDataset(records["val"], cfg, training=False)
    torch.manual_seed(3)
    out = tmp_path / "report"
    result = diagnostic(TinyModel(), train, val, out, "cpu", steps=4)
    assert result["status"] == "completed"
    assert result["test_evaluated"] is False
    assert result["benchmark_target_proven"] is False
    assert len(result["loss_history"]) == 4
    assert set(result["after"]) == {"train_subset", "validation"}
    assert (out / "validation_predictions" / "sample_000.png").exists()
    assert json.loads((out / "report.json").read_text())["train_samples"] == 2
