# Prepare the two DGX datasets

The script reads the existing extracted datasets from:

```text
/raid/workspace/AI4CV/AI4CV_CL_DGX_A100/A100_datasets/openearthmap/OpenEarthMap_wo_xBD
/raid/workspace/AI4CV/AI4CV_CL_DGX_A100/A100_datasets/landcover_ai
```

It writes RGB images and indexed masks to:

```text
/raid/workspace/RS_Dataset/
├── openearthmap/
│   ├── images/*.png
│   ├── masks/*.png
│   ├── pairs.csv
│   ├── sources.jsonl
│   └── preprocess_report.json
├── landcover_ai/
│   ├── images/*.png
│   ├── masks/*.png
│   ├── pairs.csv
│   ├── sources.jsonl
│   └── preprocess_report.json
└── configs/
    ├── openearthmap.yaml
    ├── landcover_ai.yaml
    └── experiment.yaml
```

Every output image and mask is **1024×1024**. Original files are read-only inputs.
Masks retain original class IDs, including all OpenEarthMap classes; the generated
training configurations perform the shared five-class mapping later.

## Run

Copy `prepare_datasets.py`, `requirements-preprocess.txt`, and
`configs/preprocess_dgx.yaml` into the repository (or an otherwise empty tools
directory, preserving the `configs` folder). No GPU, Torch, SAM, or PennyLane
is needed for preprocessing. Activate your existing environment, then:

```bash
python -m pip install -r requirements-preprocess.txt

# Check layouts, pair matching, dimensions and label IDs without saving.
python prepare_datasets.py --config configs/preprocess_dgx.yaml --dry-run

# Optional: save only three pairs per dataset in a separate smoke directory.
python prepare_datasets.py --config configs/preprocess_dgx.yaml \
  --limit 3 --output-root /raid/workspace/RS_Dataset/_smoke

# Prepare all paired images and masks at the requested destination.
python prepare_datasets.py --config configs/preprocess_dgx.yaml
```

The destination must be writable by your user. The script does not change
permissions or request sudo. Existing nonempty dataset outputs are rejected
unless you explicitly pass `--overwrite`. Regeneration replaces matching output
files; it does not delete old extras. The generated `pairs.csv` is authoritative
for which files the training loader uses.

## Resizing behavior

The default `resize_mode: stretch` resizes each **whole source image** to a square.
Images use bilinear interpolation, and masks use nearest-neighbor interpolation.
Mask IDs are verified before and after saving. Original dimensions and source
paths are retained in `sources.jsonl`; output PNGs do not retain georeferencing.

Resizing a very large scene can remove small roads/buildings. This implements
whole-image resizing as requested, not 1024-pixel tiling. Resizing a rectangular
scene to a square also changes its aspect ratio. For aspect-preserving resizing:

```bash
python prepare_datasets.py --config configs/preprocess_dgx.yaml \
  --resize-mode letterbox --output-root /raid/workspace/RS_Dataset_letterbox
```

Letterbox pads the bottom/right with black image pixels and mask ID 255 (ignore).
Generated configurations include the appropriate ignore mapping.

## Folder discovery and unusual releases

Supported automatic layouts include flat `images/` + `masks/` or `labels/`, and
nested `<region>/images/` + `<region>/labels/`. Region path components are preserved
in pair IDs; a stable filename hash prevents collisions after flattening.
Edit `image_dirs`, `mask_dirs`, and filename suffixes in the preprocessing YAML
if the folder names differ. Compressed archives must already be extracted.

Unmatched or ambiguous files cause a clear error. If a release contains unlabeled
images (for example, an image-only test partition), explicitly opt to omit them:

```bash
python prepare_datasets.py --config configs/preprocess_dgx.yaml \
  --allow-unlabeled-images
```

The omitted paths are listed in `preprocess_report.json`; they are not fabricated
into masks. If the download also has masks without corresponding images, use
`--paired-only` to explicitly process the intersection of matching image/mask
IDs. This excludes both kinds of unmatched file, prints counts, and records
their full paths under `skipped_unlabeled_images` and `skipped_orphan_masks` in
the report. Source files are never deleted. Verify these exclusions are acceptable
for your experiment; repairing an incomplete download is another option.
Without `--paired-only`, orphan masks remain an error. Unclassified raster files also
cause an error; a source manifest can explicitly select intended samples.

Inputs must be uint8 RGB images and integer-ID masks. Palette PNG masks retain
their indices. Three identical grayscale mask channels can be collapsed to one.
True RGB-color masks, multispectral rasters, and 16-bit imagery require an explicit
conversion profile; the script will stop rather than guess a conversion.

## Preserve official splits and scene groups

Organizing images and masks is separate from assigning train/validation/test.
The supplied configuration reads OpenEarthMap's `train_available.txt`,
`val_available.txt`, and `test_available.txt`, and LandCover.ai's `train.txt`,
`val.txt`, and `test.txt`. Paths are relative to each source directory. Entries
may be unique scene stems, filenames, or relative image paths. Every paired
scene must be assigned once, and no scene/group can cross partitions.

**LandCover.ai split lists may name smaller tiles while the input images are
whole scenes.** The script detects tile-style references and stops. It never
assigns a whole scene to a split based on one of its tiles. If you want new
scene-safe 70/15/15 partitions for resized whole scenes, explicitly use:

```bash
python prepare_datasets.py --config configs/preprocess_dgx.yaml \
  --dataset landcover_ai --split-policy scene
```

For OpenEarthMap, retain matching source lists using the default policy:

```bash
python prepare_datasets.py --config configs/preprocess_dgx.yaml --dataset openearthmap
```

Running one dataset at a time regenerates `configs/experiment.yaml` with only that
invocation's selected task(s). To combine independently prepared datasets, use
the repository's `configs/experiments/dgx_prepared.yaml`, which references both
prepared YAML files in OpenEarthMap → LandCover.ai order.

Alternatively, preserve chosen official scene splits through a CSV: remove the
dataset's `split_files` setting and set `source_manifest`:

```csv
id,image,mask,group,split
scene_a,/data/images/a.tif,/data/masks/a.tif,scene_a,train
scene_b,/data/images/b.tif,/data/masks/b.tif,scene_b,val
scene_c,/data/images/c.tif,/data/masks/c.tif,scene_c,test
```

Image/mask paths may also be relative to the CSV. With explicit splits, every row
must have train/val/test and source groups cannot cross partitions. For pre-cut
tiles, set their original scene as the group. Automatic discovery also supports
an optional `group_regex` with a named `group` capture.

Without supplied partitions (or with `--split-policy scene`), the generated training config uses seeded group
splits of 70/15/15. Each source image is a group by default. At least three groups
are needed for those splits. A limited preprocessing smoke run is not a research
dataset and may not contain adequate groups for training.

## Use the prepared data in QCL_EWC

The generated configs disable tiling and set the loader size to 1024×1024.
This prevents the project's old 512×512 defaults from shrinking/retiling the data.
The model still handles SAM's RGB normalization internally.

From the project root, with its training dependencies installed:

```bash
python validate_data.py --experiment /raid/workspace/RS_Dataset/configs/experiment.yaml

python train.py --config /raid/workspace/RS_Dataset/configs/experiment.yaml \
  --sam-checkpoint /absolute/path/to/sam_vit_b_01ec64.pth \
  --device cuda --output outputs/dgx_1024
```

The exact DGX directory contents have not been inspected remotely. Run the dry
check first; if it reports a layout mismatch, adjust the discovery configuration
or supply a source manifest. The preprocessing implementation is tested with
synthetic TIFF inputs and the existing project loader.
