import torch
import torch.nn.functional as F


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

    def update(self, logits, target):
        pred = logits.argmax(1).detach().cpu()
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
        return {"iou": [number(x) for x in iou], "miou": number(iou.nanmean()),
                "foreground_miou": number(iou[1:].nanmean()), "dice": number(dice.nanmean()),
                "confusion": self.matrix.tolist()}


@torch.no_grad()
def evaluate(model, loader, device, classes, ignore_index=255):
    model.eval()
    metric = Confusion(classes, ignore_index)
    for batch in loader:
        metric.update(model(batch["image"].to(device)), batch["mask"])
    return metric.result()
