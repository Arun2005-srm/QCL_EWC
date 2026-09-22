import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def dilate(mask, radius):
    """Square dilation, using separable pooling to avoid a large 2-D kernel."""
    x = mask.float()
    x = F.max_pool2d(x, (1, 2 * radius + 1), stride=1, padding=(0, radius))
    return F.max_pool2d(x, (2 * radius + 1, 1), stride=1, padding=(radius, 0)).bool()


def inner_boundary(mask, radius):
    # Outside the image is background, so truncated objects have boundaries too.
    outside = F.pad((~mask).float(), (radius,) * 4, value=1)
    eroded_complement = dilate(outside.bool(), radius)[..., radius:-radius, radius:-radius]
    return mask & eroded_complement


def segmentation_loss(logits, masks, ignore_index=255):
    valid = masks != ignore_index
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, masks, ignore_index=ignore_index)
    targets = F.one_hot(masks.masked_fill(~valid, 0), logits.shape[1]).permute(0, 3, 1, 2)
    probabilities = logits.softmax(1) * valid[:, None]
    targets = targets * valid[:, None]
    dims = (0, 2, 3)
    dice = (2 * (probabilities * targets).sum(dims) + 1e-6) / (probabilities.sum(dims) + targets.sum(dims) + 1e-6)
    return ce + (1 - dice.mean())


class Confusion:
    def __init__(self, classes, ignore_index=255):
        self.classes, self.ignore_index = classes, ignore_index
        self.matrix = torch.zeros(classes, classes, dtype=torch.int64)
        self.boundary_intersection = torch.zeros(classes, dtype=torch.int64)
        self.boundary_union = torch.zeros(classes, dtype=torch.int64)

    def update(self, logits, target):
        pred_device = logits.detach().argmax(1)
        target_device = target.detach().to(pred_device.device)
        radius = max(1, round(.02 * (target.shape[-2] ** 2 + target.shape[-1] ** 2) ** .5))
        # Ignore pixels and their boundary-width neighbourhood: void edges must
        # not become artificial class boundaries. Image edges remain valid.
        safe = ~dilate((target_device == self.ignore_index)[:, None], radius)
        for c in range(self.classes):
            pb = inner_boundary((pred_device == c)[:, None], radius) & safe
            tb = inner_boundary((target_device == c)[:, None], radius) & safe
            self.boundary_intersection[c] += (pb & tb).sum().cpu()
            self.boundary_union[c] += (pb | tb).sum().cpu()
        pred = pred_device.cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index
        self.matrix += torch.bincount(self.classes * target[valid] + pred[valid], minlength=self.classes**2).reshape(self.classes, self.classes)

    def result(self):
        matrix = self.matrix.double()
        tp = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - tp
        denominator = matrix.sum(0) + matrix.sum(1)
        iou = torch.where(union > 0, tp / union, torch.nan)
        dice = torch.where(denominator > 0, 2 * tp / denominator, torch.nan)
        number = lambda x: float(x) if torch.isfinite(x) else None
        boundary = torch.where(self.boundary_union > 0,
                               self.boundary_intersection.double() / self.boundary_union, torch.nan)
        return {"iou": [number(x) for x in iou], "miou": number(iou.nanmean()),
                "foreground_miou": number(iou[1:].nanmean()), "dice": number(dice.nanmean()),
                "pixel_accuracy": number(tp.sum() / matrix.sum()),
                "dice_per_class": [number(x) for x in dice],
                "boundary_iou": [number(x) for x in boundary],
                "mean_boundary_iou": number(boundary.nanmean()),
                "confusion": self.matrix.tolist()}


def metric_text(metrics):
    labels = {"loss": "loss", "pixel_accuracy": "acc", "dice": "Dice",
              "miou": "mIoU", "foreground_miou": "fg-mIoU", "mean_boundary_iou": "bIoU"}
    return " | ".join(f"{label}={metrics[key]:.4f}" if metrics.get(key) is not None
                      else f"{label}=N/A" for key, label in labels.items())


@torch.no_grad()
def evaluate(model, loader, device, classes, ignore_index=255, description="Evaluate"):
    model.eval()
    metric = Confusion(classes, ignore_index)
    loss_sum, samples = 0., 0
    with tqdm(loader, desc=description, unit="batch", dynamic_ncols=True) as progress:
        for batch in progress:
            mask = batch["mask"].to(device)
            logits = model(batch["image"].to(device))
            metric.update(logits, mask)
            if (mask != ignore_index).any():
                count = len(mask)
                loss_sum += float(segmentation_loss(logits, mask, ignore_index)) * count
                samples += count
            progress.set_postfix(loss=f"{loss_sum / samples:.4f}" if samples else "N/A")
    result = metric.result()
    result["loss"] = loss_sum / samples if samples else None
    return result
