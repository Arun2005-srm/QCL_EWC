# QCL-EWC: configuration-driven continual segmentation

A universal **paired-raster data pipeline** for remote-sensing semantic segmentation, with a SAM ViT-B + shared quantum alignment + online EWC research implementation. Add a compatible dataset by writing a YAML configuration; the model never branches on dataset names.

The data interface follows the organization of [skin-lesion-segmentation-refactored](https://github.com/Arun2005-srm/skin-lesion-segmentation-refactored), extended to multiclass masks, scene-aware tiling, and ordered continual tasks. The implementation is independent; the reference repository is unchanged.

## Quick start: just the data pipeline

Python 3.10 or later. Run commands from the repository root.

```bash
python -m venv .venv
# Linux / DGX
source .venv/bin/activate
# Windows PowerShell instead: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python smoke.py --device cpu
python -m pytest tests/test_data_pipeline.py -q
```

The smoke test creates temporary image/mask pairs, validates all scenes, builds train/validation/test loaders, and runs a small segmentation backward/optimizer step. It needs no dataset download, SAM checkpoint, or PennyLane. It is explicitly a data test, not a SAM performance test.

## Configure a dataset

Copy `configs/datasets/example_folders.yaml`, edit its paths and raw label mapping, then run:

```bash
python validate_data.py --config configs/datasets/my_dataset.yaml
```

An example complete configuration:

```yaml
defaults: ../default.yaml
dataset:
  name: my_dataset
  images:
    root: /data/my_dataset/images
    extensions: [.tif, .png, .jpg]
    scale: 255.0
  masks:
    root: /data/my_dataset/masks
    extensions: [.tif, .png]
  pairing:
    method: stem
    image_suffix: ""
    mask_suffix: _mask
  classes: [other, building, tree_woodland, water, road]
  labels:
    encoding: index
    ignore_index: 255
    mapping: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 255: 255}
```

`image_001.tif` pairs with `image_001_mask.png` in this example. Paths are relative to the YAML file that defines them, independent of the current working directory. Absolute paths, `~`, and environment variables such as `${DATA_ROOT}` are supported. Unset environment variables fail early.

Defaults merge recursively; label `mapping` and `colors` dictionaries replace their inherited equivalents completely. Pairing is strict: duplicate IDs, unmatched images, orphan masks, unknown labels, mismatched dimensions, and unsupported channel layouts raise errors.

### Folder layouts, masks, and channels

- Recursive discovery is enabled by default. Use `pairing.method: relative` when image and mask trees share nested folders and filenames repeat between folders.
- `encoding: index` preserves grayscale/palette integer IDs. Specify every raw ID in `mapping`, including void IDs. Targets are zero-based indices into `classes` or `ignore_index`.
- `encoding: rgb` uses an explicit `colors` dictionary such as `"255,0,0": 1`; see `example_rgb.yaml`. This handles RGB PNG masks and RGB TIFF arrays. Palette TIFF color expansion is not supported.
- `encoding: binary` applies `threshold` and optional `ignore_values`; see `example_binary.yaml`. No automatic conversion of multiclass data to binary occurs.
- RGB images are scaled by `images.scale`, default 255. For 16-bit inputs set an appropriate scale explicitly. For multiband TIFFs specify three zero-based `images.bands`, for example `[2, 1, 0]`. Channel-first TIFF layouts identified as `SYX`/`CYX` are supported. Unsupported stacks fail rather than silently selecting a page.
- The pipeline handles paired raster images/masks, not vector annotations, arbitrary HDF5/NetCDF stacks, or unregistered modalities. GeoTIFF georeferencing is not reprojected; pixel coordinates and original scene dimensions are retained.

### Splits and leakage prevention

**Scenes are split before tiles are created.** By default each source image is one group, with seeded 70/15/15 train/validation/test assignment. Ratios approximate the number of groups, not pixel counts. At least one group is allocated to each positive-ratio partition.

For already tiled datasets, capture their common source scene explicitly:

```yaml
dataset:
  grouping:
    regex: '^(?P<group>.+)_tile_[0-9]+$'
  split:
    strategy: group
```

Prefer official split lists when available. Convert them into a manifest with paths relative to the CSV location (or absolute paths):

```csv
id,image,mask,group,split
a,images/a.tif,masks/a.tif,scene_a,train
b,images/b.tif,masks/b.tif,scene_b,val
c,images/c.tif,masks/c.tif,scene_c,test
```

Then configure:

```yaml
dataset:
  manifest: /data/my_dataset/splits.csv
  split:
    strategy: manifest
```

The manifest takes precedence over folder discovery. `group` defaults to `id` if omitted. Do not omit source groups for pre-cut tiles. A group spanning multiple splits is rejected, even with the `random` split strategy. The validator saves a reusable `splits.csv`, `resolved.yaml`, and `report.json` under `outputs/data_validation/task_XX/`.

### Tiling, transforms, and tensor contract

Default settings create 512×512 tiles and resize to 512×512 model inputs. Set `tiling.enabled: false` to resize whole input images. Edge tiles are padded with image zeros and ignored mask pixels. Image interpolation is bilinear; mask interpolation is nearest-neighbor. Flips apply identically to images and masks and only during training.

```python
from data import load_config, build_dataloaders

cfg = load_config("configs/datasets/openearthmap.yaml")
loaders = build_dataloaders(cfg["dataset"], cfg["data_loader"])
batch = next(iter(loaders["train"]))
# image: float32 [B,3,H,W], RGB [0,1] with default normalization=null
# mask: int64 [B,H,W], class indices or ignore_index
# id, scene_id, group, dataset_id: lists of strings
# window: [B,4], source top/left/height/width before resizing
# original_size and valid_size: [B,2]
logits = model(batch["image"])
```

Metadata describes the source window, not augmented coordinates. Evaluation disables augmentation. Optional mean/std normalization is supported for other models; the SAM runner requires `normalization: null` and performs SAM's own preprocessing.

TIFF decoding caches **one full source scene per worker**, not the whole dataset. For very large rasters, supply pre-cut tiles with source groups and use `num_workers: 0` initially. This is not an out-of-core geospatial reader. Overlapping tiles repeat pixels in reported tile-level metrics; use the default nonoverlapping stride for experiments. Whole-scene stitching is not implemented.

## Continual dataset configurations

`configs/experiments/oem_landcoverai.yaml` lists dataset YAMLs in training order. New compatible datasets only require another YAML and another list entry. Every task must share the same ordered `classes` and `ignore_index`.

```bash
python validate_data.py --experiment configs/experiments/oem_landcoverai.yaml
```

OpenEarthMap and LandCover.ai configurations use:

| Shared ID | Class | OpenEarthMap raw ID | LandCover.ai raw ID |
|---|---|---|---|
| 0 | Other/background | 1, 2, 3, 7 | 0 |
| 1 | Building | 8 | 1 |
| 2 | Tree/woodland | 5 | 2 |
| 3 | Water | 6 | 3 |
| 4 | Road | 4 | 4 |
| 255 | Ignore | 0 | explicitly map any release-specific void ID |

These are templates for the original datasets, not every repackaged release. Validate raw IDs before research runs. Tree and woodland annotation definitions are not identical, nor are all road/background conventions. This is an approximate cross-dataset harmonization, not an exact ontology equivalence. Sources: [OpenEarthMap](https://arxiv.org/abs/2210.10732), [LandCover.ai](https://arxiv.org/abs/2005.02264).

## SAM, quantum alignment, and EWC

```text
RGB tile → frozen SAM ViT-B encoder → shared residual alignment
         → frozen SAM decoder + shared learned class prompts → five-class logits

Alignment: 4×4 pooled features → four angles → four-qubit circuit
           → Pauli-Z expectations → 256 channels → spatial upsampling + residual
```

Two trainable rotation/entanglement layers use PennyLane `default.qubit` with analytic expectations and backpropagation. The small quantum circuit runs on CPU; tensor transfers preserve gradients when the rest of the model uses CUDA. Quantum simulation is not automatically accelerated by an A100.

SAM parameters are frozen. The alignment projections, circuit angles, and shared class prompts are trainable. No task router, per-task adapter, or ground-truth box/point prompt is used. The learned class prompts are a proposed semantic extension to the [CA-SAM idea](https://arxiv.org/abs/2511.17201), not a reproduction of its medical segmentation protocol or a proven quantum advantage.

Training uses cross-entropy + soft multiclass Dice + online EWC. The diagonal **empirical** Fisher averages squared individual valid-pixel log-likelihood gradients using current-task training samples. It is not the quantum Fisher, a squared average gradient, or a Dice-loss gradient. Fisher sampling is image-balanced; its budgets are configurable. Default online accumulation has decay 1 and a single latest anchor, keeping EWC state constant in size across tasks. This differs from retaining a separate original-EWC penalty per task.

`ewc.scope: all` protects all trainable parameters. `scope: quantum` protects only circuit angles and is an ablation. `strength: 0` disables consolidation and the penalty. No prior-task examples are replayed. Earlier validation/test partitions remain available for reporting only.

## DGX setup and smoke runs

Use one A100 first. Install the mutually compatible CUDA-enabled PyTorch and torchvision versions appropriate to the DGX driver from [PyTorch's official selector](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements-model.txt
python -m pip check
python smoke.py --device cuda
python smoke_model.py --device cuda
```

The official SAM source is pinned to a commit. Download the **ViT-B** checkpoint from the [official SAM checkpoint list](https://github.com/facebookresearch/segment-anything#model-checkpoints), and place it outside Git, for example `checkpoints/sam_vit_b_01ec64.pth`.

```bash
python smoke_model.py --device cuda \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --two-task \
  --output outputs/dgx_smoke/report.json
```

This verifies real pretrained SAM forward/backward gradients, two small synthetic task runs, Fisher consolidation, held-out evaluation, and checkpoint saving. It does **not** measure remote-sensing performance. A checkpoint-free architecture check is also available:

```bash
python smoke_model.py --device cpu --random-sam --two-task
```

The command explicitly reports random weights; it never silently substitutes them for missing pretrained weights. Synthetic smoke assets are saved under the output directory. Do not use their checkpoints as research-trained models.

## Train, compare, resume, and evaluate

```bash
python train.py --config configs/experiments/oem_landcoverai.yaml \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --device cuda \
  --output outputs/quantum_ewc

python train.py --config configs/experiments/oem_landcoverai.yaml \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --device cuda \
  --output outputs/quantum_ewc --resume outputs/quantum_ewc/last.pt

python evaluate.py --checkpoint outputs/quantum_ewc/task_01.pt \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --device cuda
```

`evaluate.py` uses saved scene partitions. It writes JSON metrics and image/ground-truth/prediction panels. The default palette is gray=other, red=building, green=tree/woodland, blue=water, yellow=road; white denotes ignored pixels.

Use distinct output directories for these four comparisons:

| Comparison | Additional train arguments |
|---|---|
| Quantum + EWC | `--alignment quantum --ewc-strength 100` |
| Quantum, no EWC | `--alignment quantum --ewc-strength 0` |
| Classical + EWC | `--alignment classical --ewc-strength 100` |
| Classical, no EWC | `--alignment classical --ewc-strength 0` |

The classical baseline preserves the four-value bottleneck and coarse grid; it is not exactly parameter-count matched. Report parameter counts and timing. Repeat with `--seed 42`, `43`, `44`; scene split seeds remain fixed so the compared data partitions are identical. EWC strength 100 and five epochs per task are starting settings, not tuned research recommendations; tune on validation data only.

Each epoch saves `last.pt`; each task saves `task_XX.pt` after restoring its best validation-mIoU parameters and consolidating EWC. The next task receives a new optimizer. Resume is supported at epoch boundaries with `num_workers: 0`; changed configurations, SAM checkpoint hashes, or discovered scene partitions are rejected. Dataset file contents are not hashed: keep the underlying files unchanged when resuming. The pretrained SAM weights are external and are not duplicated in training checkpoints.

`metrics.json` includes per-class IoU, five-class and foreground mIoU, Dice, confusion matrices, the lower-triangular task performance matrix, old-task forgetting, epoch history, parameter count, elapsed time, and peak GPU allocation. Classes with no union are omitted from mIoU rather than scored as perfect. Forgetting is the best earlier mIoU minus final mIoU; negative values represent improvement. No accuracy target or quantum advantage is asserted by the smoke tests.

## Layout and verification

```text
configs/        Dataset mappings, shared defaults, ordered experiment
data/           Config, pairing, splitting, raster I/O, tiling, loaders
models/         Shared quantum/classical alignment and frozen SAM integration
continual/      EWC, losses, metrics, checkpoint helpers, visualizations
tests/          Data contracts, Fisher correctness, resume, quantum gradients
validate_data.py / smoke.py / smoke_model.py / train.py / evaluate.py
```

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Quantum tests skip if optional PennyLane dependencies are absent. Data-only commands do not import SAM or PennyLane. Generated outputs, datasets, virtual environments, and checkpoints are excluded from Git. Nothing is pushed automatically.
