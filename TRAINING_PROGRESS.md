# Training progress and metrics

Train, validation, and test loops show tqdm batch counts, elapsed time, and ETA.
Training bars show cumulative segmentation loss, pixel accuracy, and mIoU.
Epoch summaries include train/validation loss, pixel accuracy, macro Dice,
mIoU, foreground mIoU (excluding class 0), mean Boundary IoU, and named per-class IoU.
All scores are fractions from 0 to 1. Undefined class scores are null/N/A;
classes with zero union are excluded from macro averages.

Training loss is CE plus soft Dice, averaged by batch sample count over batches
with valid labels. EWC penalty and total optimization loss are reported separately.
Validation loss is CE plus soft Dice without EWC, with the same averaging convention.
Training metrics use predictions collected during optimization with training
augmentation, not a separate evaluation of the final epoch model. Validation
uses evaluation mode. Accuracy is valid-pixel accuracy, not image accuracy.
Region scores aggregate the confusion matrix across the full split.
The legacy top-level history `loss` remains the mean batch total optimization loss.

Boundary IoU follows the inner boundary-band definition in
https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py
using a square erosion radius max(1, round(0.02 * image diagonal)); for a
1024 x 1024 mask this is 29 pixels. Image edges count as boundaries.
For semantic masks, intersections and unions are accumulated per class across
the split, then averaged over classes with nonzero boundary union, including
class 0. This is a semantic adaptation, not COCO instance AP. Pixels within
that radius of an ignored target pixel are excluded from boundary scoring.
Boundary IoU is a reporting metric, not an extra training loss.

Each epoch saves history.json and last.pt. Final metrics.json includes history,
per-class Dice/IoU/Boundary IoU, test metrics, and continual-learning forgetting.
Best-checkpoint selection still uses validation mIoU. Fisher consolidation
prints its phase and sampling budget. Boundary computation adds some overhead.

## Apply on DGX

Upload outputs/training_progress.zip from your local computer, then on DGX:

```bash
cd /raid/workspace/QCL_EWC/QCL_EWC
unzip -o training_progress.zip
source .venv/bin/activate
python -m pip install 'tqdm>=4.66,<5'
```

Use the same training command. Updates take effect when a new Python process
starts. If a run already has last.pt, resume with the same config and output:
`--resume outputs/dgx_oem_then_landcover/last.pt`. Resume restores the last saved
epoch boundary, not a partially completed epoch. Existing histories do not gain
metrics that were not collected by the old code.
