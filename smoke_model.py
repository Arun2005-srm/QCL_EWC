"""Smoke the real quantum circuit, optionally with the actual SAM ViT-B."""
import argparse
import json
from pathlib import Path
import time

import torch

from continual.metrics import segmentation_loss
from models.alignment import SpatialAlignment
from models.sam_segmenter import UniversalSAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-checkpoint", help="Official SAM ViT-B checkpoint")
    parser.add_argument("--random-sam", action="store_true", help="Test full SAM architecture with random weights; not a pretrained test")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--decoder", choices=["prompt", "semantic"], default="prompt")
    parser.add_argument("--two-task", action="store_true", help="Also train/evaluate two tiny synthetic tasks and save EWC/checkpoints")
    parser.add_argument("--visualize", action="store_true", help="Save labeled predictions after each task (requires --two-task)")
    parser.add_argument("--output", default="outputs/model_smoke.json")
    args = parser.parse_args()
    if args.sam_checkpoint and args.random_sam:
        parser.error("Choose a pretrained checkpoint OR --random-sam")
    if args.two_task and not (args.sam_checkpoint or args.random_sam):
        parser.error("--two-task requires --sam-checkpoint or --random-sam")
    if args.visualize and not args.two_task:
        parser.error("--visualize requires --two-task")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.manual_seed(42)
    started = time.perf_counter()
    full = bool(args.sam_checkpoint or args.random_sam)
    if full:
        sam = None
        if args.random_sam:
            from segment_anything import sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=None)
        model = UniversalSAM(5, args.sam_checkpoint, sam=sam, decoder=args.decoder,
                             gated=args.decoder == "semantic").to(args.device)
        image = torch.rand(1, 3, 64, 64, device=args.device)
        mask = torch.randint(0, 5, (1, 64, 64), device=args.device)
        logits = model(image)
        loss = segmentation_loss(logits, mask)
    else:
        model = SpatialAlignment().to(args.device)
        features = torch.randn(1, 256, 8, 8, device=args.device)
        logits = model(features)
        loss = logits.square().mean()
    loss.backward()
    grads = {n: float(p.grad.abs().sum().cpu()) if p.grad is not None else None for n, p in model.named_parameters() if p.requires_grad}
    if any(value is None or not 0 < value < float("inf") for value in grads.values()):
        raise AssertionError(f"Missing, zero or non-finite trainable gradients: {grads}")
    if full:
        assert all(p.grad is None and not p.requires_grad for p in model.sam.parameters())
    result = {"status": "passed", "test": "pretrained SAM ViT-B" if args.sam_checkpoint else ("random-weight SAM ViT-B architecture" if args.random_sam else "quantum alignment only"),
              "device": args.device, "decoder": args.decoder, "shape": list(logits.shape), "loss": float(loss.detach().cpu()),
              "gradient_l1": grads, "elapsed_seconds": time.perf_counter() - started}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.two_task:
        import csv
        from data import load_config
        from smoke import make_fixture
        from train import run_training
        from continual.experiment import file_hash
        tasks = []
        for index in range(2):
            root = output.parent / "smoke_stream_data" / str(index)
            cfg = load_config(make_fixture(root))
            manifest = root / "selected.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "image", "mask", "group", "split"])
                for scene, split in enumerate(("train", "val", "test")):
                    writer.writerow([f"scene_{scene}", f"images/scene_{scene}.png", f"masks/scene_{scene}_mask.png", f"scene_{scene}", split])
            cfg["dataset"].update(name=f"synthetic_{index}", manifest=str(manifest.resolve()), split={"strategy": "manifest"})
            cfg["dataset"]["tiling"]["enabled"] = False
            tasks.append(cfg)
        experiment = {"model": {"kind": "quantum", "qubits": 4, "depth": 2, "grid": 4,
                                "decoder": args.decoder, "gated": args.decoder == "semantic"},
                      "training": {"epochs_per_task": 1, "seed": 42},
                      "ewc": {"strength": 100., "fisher_images": 1, "fisher_pixels_per_image": 1}}
        stream = run_training(experiment, tasks, model, output.parent / "smoke_stream", torch.device(args.device),
                              file_hash(args.sam_checkpoint) if args.sam_checkpoint else "random-sam-smoke")
        result["two_task_matrix_shape"] = [len(row) for row in stream["matrix"]]
        if args.visualize:
            from data import build_records
            from data.dataset import SegmentationDataset
            from continual.visualization import save_smoke_comparison
            cfg = tasks[0]["dataset"]
            dataset = SegmentationDataset(build_records(cfg)["test"], cfg)
            states = [torch.load(output.parent / "smoke_stream" / f"task_{i:02d}.pt", map_location="cpu", weights_only=True)["model"] for i in range(2)]
            preview = output.parent / "smoke_comparison.png"
            save_smoke_comparison(model, dataset, args.device, states, stream["matrix"], preview,
                                  pretrained=bool(args.sam_checkpoint))
            result["visualization"] = str(preview.resolve())
        result["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
