"""Save image / label / prediction panels without plotting dependencies."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


@torch.no_grad()
def save_predictions(model, dataset, device, destination, limit=3):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    # Fixed five-class legend in README; additional generic classes use a seeded palette.
    palette = np.random.default_rng(42).integers(40, 240, (256, 3), dtype=np.uint8)
    palette[:5] = [[60, 60, 60], [220, 60, 60], [40, 170, 70], [40, 100, 230], [230, 190, 40]]
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        prediction = model(sample["image"][None].to(device))[0].argmax(0).cpu().numpy()
        mask = sample["mask"].numpy()
        valid = mask != dataset.cfg["labels"].get("ignore_index", 255)
        image = (sample["image"].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        truth = palette[np.where(valid, mask, 0) % 256].copy()
        pred = palette[prediction % 256].copy()
        truth[~valid], pred[~valid] = 255, 255
        Image.fromarray(np.concatenate([image, truth, pred], axis=1)).save(destination / f"sample_{index:03d}.png")


@torch.no_grad()
def save_smoke_comparison(model, dataset, device, states, matrix, destination, pretrained=False):
    """Show the same held-out example after each task; restore the caller's model."""
    sample = dataset[0]
    palette = np.array([[60, 60, 60], [220, 60, 60], [40, 170, 70],
                        [40, 100, 230], [230, 190, 40]], dtype=np.uint8)
    original_state, original_mode = model.trainable_state(), model.training
    image = (sample["image"].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    mask = sample["mask"].numpy()
    valid = mask != dataset.cfg["labels"].get("ignore_index", 255)
    def colorize(labels):
        colors = palette[np.where(valid, labels, 0)].copy()
        colors[~valid] = 255
        return colors
    panels = [image, colorize(mask)]
    try:
        model.eval()
        for state in states:
            model.load_trainable_state(state)
            prediction = model(sample["image"][None].to(device))[0].argmax(0).cpu().numpy()
            panels.append(colorize(prediction))
    finally:
        model.load_trainable_state(original_state)
        model.train(original_mode)
    def font(size):
        for candidate in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
        return ImageFont.load_default()
    canvas = Image.new("RGB", (1280, 660), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((32, 24), "SAM + quantum alignment + EWC | smoke run", fill="#18243a", font=font(30))
    initialization = "Pretrained SAM" if pretrained else "Random SAM weights"
    draw.text((32, 70), f"{initialization} • Synthetic data • {device.upper()} • 1 epoch per task", fill="#526078", font=font(20))
    labels = ["Synthetic input", "Ground truth", "After task 1", "After task 2"]
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = 32 + index * 312
        draw.text((left, 121), label, fill="#18243a", font=font(22))
        canvas.paste(Image.fromarray(panel).resize((280, 280), Image.Resampling.NEAREST), (left, 158))
    for index, (name, color) in enumerate(zip(["Other", "Building", "Tree/woodland", "Water", "Road"], palette)):
        left = 32 + index * 245
        draw.rectangle((left, 466, left + 20, 486), fill=tuple(color.tolist()))
        draw.text((left + 30, 462), name, fill="#18243a", font=font(19))
    first, last = matrix[0][0]["miou"], matrix[-1][0]["miou"]
    draw.text((32, 512), f"Task 1 held-out mIoU: {first:.3f} after task 1  →  {last:.3f} after task 2", fill="#18243a", font=font(22))
    draw.text((32, 554), "Pipeline check only: these predictions are not evidence of segmentation quality or reduced forgetting.", fill="#526078", font=font(19))
    draw.text((32, 592), "Both tiny tasks use the same synthetic distribution. Colors represent predicted class IDs.", fill="#526078", font=font(19))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
