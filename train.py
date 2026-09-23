"""Sequential dataset training for shared SAM alignment with optional EWC."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm

from continual.ewc import OnlineEWC
from continual.experiment import read_experiment, file_hash, atomic_save
from continual.metrics import segmentation_loss, evaluate, Confusion, metric_text
from data import build_records, build_dataloaders, save_manifest
from data.dataset import SegmentationDataset
from models.sam_segmenter import UniversalSAM


def validate_continuation(saved, specs):
    """Allow only appending tasks after a completed run, with its protocol intact."""
    old = saved["specs"]
    count = len(old["tasks"])
    if saved["epoch"] != 0 or saved["task"] != count:
        raise ValueError("--continue-from requires a completed final task checkpoint, after Fisher consolidation")
    if len(specs["tasks"]) <= count or specs["tasks"][:count] != old["tasks"]:
        raise ValueError("Continuation must append tasks without changing completed dataset configurations")
    if specs["sam_sha256"] != old["sam_sha256"]:
        raise ValueError("Continuation requires the same SAM checkpoint")
    old_settings = {k: v for k, v in old["experiment"].items() if k != "tasks"}
    new_settings = {k: v for k, v in specs["experiment"].items() if k != "tasks"}
    if old_settings != new_settings:
        raise ValueError("Continuation must preserve model, training, loader, and EWC settings")


def run_training(experiment, tasks, model, output, device, sam_hash, resume=None, extend=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = experiment.get("training", {})
    epochs = training.get("epochs_per_task", 5)
    seed = training.get("seed", 42)
    if epochs < 1:
        raise ValueError("epochs_per_task must be positive")
    optimizer_kind = training.get("optimizer", "adam")
    schedule = training.get("schedule", "constant")
    if optimizer_kind not in {"adam", "adamw"} or schedule not in {"constant", "cosine"}:
        raise ValueError("Use optimizer adam/adamw and schedule constant/cosine")
    if any(c.get("data_loader", {}).get("num_workers", 0) > 0
           and c.get("data_loader", {}).get("persistent_workers", False) for c in tasks):
        raise ValueError("Training requires persistent_workers=false so worker RNG resets reproducibly at epoch boundaries")
    settings = experiment.get("ewc", {})
    ewc = OnlineEWC(settings.get("strength", 100), settings.get("decay", 1), settings.get("scope", "all"))
    specs = {"experiment": experiment, "tasks": tasks, "sam_sha256": sam_hash}
    print("Active model: " + json.dumps({
        "decoder": getattr(model, "decoder_kind", type(model).__name__),
        "alignment": experiment.get("model", {}).get("kind", "quantum"),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer": optimizer_kind, "base_learning_rate": training.get("learning_rate", .001),
        "warmup_epochs": training.get("warmup_epochs", 0), "schedule": schedule,
        "epochs_per_task": epochs,
    }), flush=True)
    model.to(device)
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    start_task, start_epoch, history, matrix, best_score, best_state = 0, 0, [], [], -1., None
    if saved:
        if extend:
            validate_continuation(saved, specs)
        elif saved["specs"] != specs:
            raise ValueError("Resume configuration or SAM checkpoint differs from the saved run")
        model.load_trainable_state(saved["model"])
        ewc.load_state_dict(saved["ewc"])
        start_task, start_epoch = saved["task"], saved["epoch"]
        history, matrix = saved["history"], saved["matrix"]
        best_score, best_state = saved["best_score"], saved["best_state"]
    records = [build_records(c["dataset"]) for c in tasks]
    for i, record in enumerate(records):
        if any(not record[s] for s in ("train", "val", "test")):
            raise ValueError("Research training requires nonempty train/val/test scene splits")
        manifest = output / f"task_{i:02d}_splits.csv"
        if saved and i < len(saved["records"]):
            # Detect changed discoveries/partitions before overwriting a saved manifest.
            if saved["records"][i] != {s: [(r.sample_id, str(r.image_path), str(r.mask_path), r.group) for r in subset] for s, subset in record.items()}:
                raise ValueError("Dataset scene records or partitions changed since the checkpoint")
        save_manifest(record, manifest)
    serialized_records = [{s: [(r.sample_id, str(r.image_path), str(r.mask_path), r.group) for r in subset] for s, subset in rec.items()} for rec in records]
    (output / "resolved.yaml").write_text(yaml.safe_dump(specs, sort_keys=False), encoding="utf-8")
    loaders = [build_dataloaders(c["dataset"], c.get("data_loader"), records=r) for c, r in zip(tasks, records)]
    classes = len(tasks[0]["dataset"]["classes"])
    ignore = tasks[0]["dataset"]["labels"].get("ignore_index", 255)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(task, epoch, optimizer, generator):
        return {"specs": specs, "records": serialized_records, "model": model.trainable_state(),
                "ewc": ewc.state_dict(), "task": task, "epoch": epoch, "history": history, "matrix": matrix,
                "best_score": best_score, "best_state": best_state,
                "optimizer": optimizer.state_dict() if optimizer else None,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "loader_rng": generator.get_state() if generator else None}

    for task_index in range(start_task, len(tasks)):
        torch.manual_seed(seed + task_index)
        optimizer_cls = torch.optim.AdamW if optimizer_kind == "adamw" else torch.optim.Adam
        base_lr = training.get("learning_rate", .001)
        optimizer = optimizer_cls((p for p in model.parameters() if p.requires_grad), lr=base_lr,
                                  weight_decay=training.get("weight_decay", 0.))
        current = loaders[task_index]
        task_name = tasks[task_index]["dataset"]["name"]
        print(f"\nTask {task_index + 1}/{len(tasks)}: {task_name} | "
              + ", ".join(f"{s}: {len(current[s].dataset)} samples / {len(current[s])} batches"
                          for s in ("train", "val", "test")), flush=True)
        epoch_start = start_epoch if task_index == start_task else 0
        if saved and task_index == start_task:
            if saved["optimizer"]:
                optimizer.load_state_dict(saved["optimizer"])
            if epoch_start > 0:
                torch.set_rng_state(saved["torch_rng"])
                if device.type == "cuda" and saved["cuda_rng"]:
                    torch.cuda.set_rng_state_all(saved["cuda_rng"])
                if saved["loader_rng"] is not None:
                    current["train"].generator.set_state(saved["loader_rng"])
        for epoch in range(epoch_start, epochs):
            warmup = training.get("warmup_epochs", 0)
            if warmup < 0 or warmup >= epochs:
                raise ValueError("warmup_epochs must be >=0 and less than epochs_per_task")
            if epoch < warmup:
                lr = base_lr * (epoch + 1) / warmup
            elif schedule == "cosine":
                fraction = (epoch - warmup) / max(1, epochs - warmup - 1)
                lr = base_lr * (.05 + .95 * .5 * (1 + math.cos(math.pi * fraction)))
            else:
                lr = base_lr
            for group in optimizer.param_groups:
                group["lr"] = lr
            print(f"Epoch {epoch + 1}/{epochs}: learning_rate={lr:.8g}; "
                  "bar metrics are cumulative training predictions, not validation scores", flush=True)
            model.train()
            loss_sum, steps = 0., 0
            seg_sum, penalty_sum, samples = 0., 0., 0
            train_metric = Confusion(classes, ignore)
            progress = tqdm(current["train"], desc=f"Train T{task_index + 1} E{epoch + 1}/{epochs}",
                            unit="batch", dynamic_ncols=False, ncols=160)
            for batch in progress:
                if not (batch["mask"] != ignore).any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["image"].to(device))
                mask = batch["mask"].to(device)
                seg_loss = segmentation_loss(logits, mask, ignore)
                penalty = ewc.penalty(model)
                loss = seg_loss + penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
                steps += 1
                count = len(mask)
                samples += count
                seg_sum += float(seg_loss.detach()) * count
                penalty_sum += float(penalty.detach()) * count
                train_metric.update(logits, mask)
                running = train_metric.result()
                progress.set_postfix(loss=f"{seg_sum / samples:.4f}",
                                     acc=f"{running['pixel_accuracy']:.4f}",
                                     mIoU=f"{running['miou']:.4f}")
            progress.close()
            if not steps:
                raise ValueError("Training task has no valid labeled pixels")
            train_metrics = train_metric.result()
            train_metrics.update(loss=seg_sum / samples, ewc_penalty=penalty_sum / samples,
                                 total_loss=(seg_sum + penalty_sum) / samples)
            validation = evaluate(model, current["val"], device, classes, ignore,
                                  description=f"Val T{task_index + 1} E{epoch + 1}/{epochs}")
            score = validation["miou"]
            if score is None:
                raise ValueError("Validation split has no valid labeled pixels")
            if score > best_score:
                best_score, best_state = score, model.trainable_state()
            entry = {"task": task_index, "epoch": epoch + 1, "learning_rate": lr, "loss": loss_sum / steps,
                     "training": train_metrics, "validation": validation}
            history.append(entry)
            print(f"Epoch {epoch + 1}/{epochs} | {task_name}\n"
                  f"  train: {metric_text(train_metrics)} | EWC={train_metrics['ewc_penalty']:.4f}"
                  f" | total_loss={train_metrics['total_loss']:.4f}\n"
                  f"  val:   {metric_text(validation)}", flush=True)
            for label, metrics in (("train", train_metrics), ("val", validation)):
                values = ", ".join(f"{name}={value:.4f}" if value is not None else f"{name}=N/A"
                                   for name, value in zip(tasks[0]["dataset"]["classes"], metrics["iou"]))
                print(f"  {label} IoU: {values}", flush=True)
            atomic_save(checkpoint(task_index, epoch + 1, optimizer, current["train"].generator), output / "last.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        model.load_trainable_state(best_state)
        # Deterministic Fisher sampling from current TRAIN scenes, without flips.
        fisher_loader = DataLoader(SegmentationDataset(records[task_index]["train"], tasks[task_index]["dataset"], training=False),
                                   batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed + task_index))
        print(f"{task_name}: consolidating EWC Fisher (up to {settings.get('fisher_images', 16)} training samples)", flush=True)
        ewc.consolidate(model, fisher_loader, device, ignore, settings.get("fisher_images", 16),
                        settings.get("fisher_pixels_per_image", 4), seed + task_index)
        row = [evaluate(model, loaders[j]["test"], device, classes, ignore,
                        description=f"Test {tasks[j]['dataset']['name']}") for j in range(task_index + 1)]
        matrix.append(row)
        best_score, best_state = -1., None
        state = checkpoint(task_index + 1, 0, None, None)
        atomic_save(state, output / f"task_{task_index:02d}.pt")
        atomic_save(state, output / "last.pt")
        saved = None
    forgetting = []
    for j in range(max(0, len(matrix) - 1)):
        past = [row[j]["miou"] for row in matrix[j:-1] if row[j]["miou"] is not None]
        last = matrix[-1][j]["miou"]
        forgetting.append(max(past) - last if past and last is not None else None)
    result = {"classes": tasks[0]["dataset"]["classes"], "tasks": [c["dataset"]["name"] for c in tasks],
              "matrix": matrix, "forgetting_miou": forgetting, "history": history,
              "elapsed_seconds_this_invocation": time.perf_counter() - started,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/quantum_ewc")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--alignment", choices=["quantum", "classical"])
    parser.add_argument("--ewc-strength", type=float)
    parser.add_argument("--seed", type=int)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", help="Resume the same experiment at an epoch boundary")
    continuation.add_argument("--continue-from", help="Append new tasks after a completed run; restore model and EWC state")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; install the appropriate CUDA PyTorch build or use --device cpu")
    output = Path(args.output)
    if (output / "last.pt").exists() and not args.resume:
        parser.error("Output contains a run; use a new --output for continuation, or --resume for an existing experiment")
    experiment, tasks = read_experiment(args.config)
    if args.alignment:
        experiment.setdefault("model", {})["kind"] = args.alignment
    if args.ewc_strength is not None:
        experiment.setdefault("ewc", {})["strength"] = args.ewc_strength
    if args.seed is not None:
        experiment.setdefault("training", {})["seed"] = args.seed
        for task in tasks:
            task.setdefault("data_loader", {})["seed"] = args.seed
    seed = experiment.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    model = UniversalSAM(len(tasks[0]["dataset"]["classes"]), args.sam_checkpoint, **experiment.get("model", {}))
    run_training(experiment, tasks, model, args.output, torch.device(args.device), file_hash(args.sam_checkpoint),
                 args.resume or args.continue_from, extend=bool(args.continue_from))


if __name__ == "__main__":
    main()
