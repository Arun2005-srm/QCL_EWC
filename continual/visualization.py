"""Save image / label / prediction panels without plotting dependencies."""
from pathlib import Path

import numpy as np
from PIL import Image
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
