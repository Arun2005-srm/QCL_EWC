"""Real-data learning diagnostic, not proof of benchmark/SOTA performance.

Overfit a fixed subset of TRAIN scenes; evaluate the full validation split before
and after. Never loads the test split. No augmentation, resizing overrides, or
class removal is used to inflate the score. Uses the configured architecture.
"""
import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from continual.experiment import read_experiment, file_hash, atomic_save
from continual.metrics import segmentation_loss, evaluate
from continual.visualization import save_predictions
from data import build_records, save_manifest
from data.dataset import SegmentationDataset
from models.sam_segmenter import UniversalSAM


def diagnostic(model, train_data, val_data, output, device, steps=200, batch_size=2, lr=.001):
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("steps, batch_size, and learning rate must be positive")
    if not len(train_data) or not len(val_data):
        raise ValueError("Require nonempty training and validation datasets")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new diagnostic output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model.to(device)
    classes = len(train_data.cfg["classes"])
    ignore = train_data.cfg["labels"].get("ignore_index", 255)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(42), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)
    train_eval = DataLoader(train_data, batch_size=batch_size, num_workers=0)
    before = {"train_subset": evaluate(model, train_eval, device, classes, ignore, "Initial train"),
              "validation": evaluate(model, val_loader, device, classes, ignore, "Initial val")}
    report = {"purpose": "training-subset overfit diagnostic; NOT a benchmark result",
              "test_evaluated": False, "benchmark_target_proven": False,
              "train_samples": len(train_data), "validation_samples": len(val_data),
              "before": before, "status": "running", "steps_requested": steps}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    iterator = iter(train_loader)
    first_gradients = {}
    loss_history = []
    model.train()
    with tqdm(range(steps), desc="Train-subset diagnostic", ncols=160) as progress:
        for step in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            mask = batch["mask"].to(device)
            if not (mask != ignore).any():
                raise ValueError("Diagnostic batch contains only ignored labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            loss = segmentation_loss(logits, mask, ignore)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic loss")
            loss.backward()
            if step == 0:
                first_gradients = {n: float(p.grad.norm()) if p.grad is not None else None
                                   for n, p in model.named_parameters() if p.requires_grad}
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          1., error_if_nonfinite=True)
            optimizer.step()
            loss_history.append(float(loss.detach()))
            progress.set_postfix(loss=f"{loss_history[-1]:.4f}")
    after = {"train_subset": evaluate(model, train_eval, device, classes, ignore, "Final train"),
             "validation": evaluate(model, val_loader, device, classes, ignore, "Final val")}
    report.update(status="completed", after=after, loss_history=loss_history,
                  first_gradient_norms=first_gradients,
                  training_loss_decreased=after["train_subset"]["loss"] < before["train_subset"]["loss"],
                  train_miou_86_reached=(after["train_subset"]["miou"] or 0) >= .86,
                  validation_miou_86_reached=(after["validation"]["miou"] or 0) >= .86)
    atomic_save({"model": model.trainable_state(), "diagnostic_only": True}, output / "diagnostic_weights.pt")
    save_predictions(model, train_data, device, output / "train_predictions", limit=3)
    save_predictions(model, val_data, device, output / "validation_predictions", limit=3)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--sam-checkpoint", required=True)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--train-scenes", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    experiment, tasks = read_experiment(args.config)
    if not 0 <= args.task < len(tasks) or args.train_scenes < 1:
        p.error("Invalid task index or train-scenes count")
    torch.manual_seed(42)
    cfg = tasks[args.task]["dataset"]
    records = build_records(cfg)
    selected = sorted(records["train"], key=lambda r: r.sample_id)[:args.train_scenes]
    train_data = SegmentationDataset(selected, cfg, training=False)
    val_data = SegmentationDataset(records["val"], cfg, training=False)
    model = UniversalSAM(len(cfg["classes"]), args.sam_checkpoint, **experiment.get("model", {}))
    report = diagnostic(model, train_data, val_data, args.output, args.device,
                        args.steps, args.batch_size, args.learning_rate)
    save_manifest({"train": selected, "val": records["val"]}, Path(args.output) / "diagnostic_splits.csv")
    report.update(experiment=experiment, dataset=cfg, sam_sha256=file_hash(args.sam_checkpoint))
    (Path(args.output) / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "before", "after", "benchmark_target_proven")}, indent=2))


if __name__ == "__main__":
    main()
