# Stronger SAM + quantum segmentation baseline

This is an implemented architecture upgrade, not a measured improvement on the
real datasets or a reproduction of HQF-Net. No 85.68% mIoU claim is made.

## Architecture

`model.decoder: prompt` retains the previous model and old trainable-state keys.
`model.decoder: semantic` replaces the prompt-driven frozen SAM mask decoder
with a trainable semantic decoder. SAM ViT-B image encoder remains frozen.

RGB patches are resized/padded for SAM's standard 1024 input and normalized using
SAM preprocessing. The original RGB patch independently enters a trainable
convolutional detail encoder (four scales: half, quarter, eighth, sixteenth).
SAM's 256-channel final embedding passes through the quantum/classical alignment
module, optionally scaled by a learned sigmoid residual gate initialized near
0.12. The semantic decoder progressively upsamples and concatenates these
semantics with the four RGB detail features, then outputs one logit per class
per original patch pixel. GroupNorm avoids dependence on large batch statistics.
This is multi-scale RGB fusion, not extraction of intermediate SAM ViT layers.
For rectangular patches SAM's padded embedding area is cropped before fusion.
The quantum circuit remains four-qubit CPU simulation; there is no claim of
quantum hardware speedup. Grid 8 uses 64 circuits per image, versus 16 at grid 4.

The frozen SAM encoder can process smaller sub-batches using
`encoder_batch_size: 1`, reducing its peak attention memory without splitting
the decoder's optimization batch. This trades throughput for memory.

## Data: preserve native spatial detail

Do not train this version on whole LandCover.ai orthophotos shrunk to 1024.
The new exporter recovers original file paths from existing `sources.jsonl`,
uses the existing scene partition before cropping, decodes each source once,
then writes nonoverlapping 512 x 512 PNG tiles. Edge pixels are padded with
RGB zero and label 255. Images are not resized; label mappings are applied once.
Provenance records source files and crop coordinates for every exported tile.
Existing prepared files and original rasters are not changed. Use a new empty
output directory; a failed partial export must be inspected before rerunning.

```bash
python prepare_native_tiles.py \
  --config /raid/workspace/RS_Dataset/configs/landcover_ai.yaml \
  --sources /raid/workspace/RS_Dataset/landcover_ai/sources.jsonl \
  --output /raid/workspace/RS_Dataset/native512/landcover_ai
```

It may produce thousands of tiles and needs more disk space than 41 resized
scenes. The original TIFF paths in sources.jsonl must still exist. This preserves
your custom scene split, not an official benchmark split. Do not compare its
score directly to 85.68% without reconciling evaluation protocols.

## Start with an isolated LandCover.ai experiment

```bash
CUDA_VISIBLE_DEVICES=2 python -u train.py \
  --config configs/experiments/dgx_semantic_landcover.yaml \
  --sam-checkpoint /raid/workspace/AI4CV/models/sam_vit_b_01ec64.pth \
  --device cuda --output outputs/semantic_landcover_quantum
```

The starting configuration uses batch 2, two nonpersistent workers, 100 epochs,
AdamW (3e-4 learning rate, 1e-4 weight decay), five warmup epochs followed by cosine
decay to 5% of the base learning rate, and CE + soft Dice. These are starting
hyperparameters, not validated optimal settings. Keep batch 2 until GPU memory
has been measured with the new decoder; previous memory estimates do not apply.
GroupNorm and trainable decoder features add memory and computation.

For a classical control use the exact same command/config plus
`--alignment classical --output outputs/semantic_landcover_classical` (supply
only one --output). It retains the same decoder and gated residual structure;
the circuit and MLP bottlenecks are not exactly parameter-count matched.
Repeat with seeds 42, 43, 44; tune only on validation data. Inspect masks as well
as per-class metrics. Single-task config disables EWC; EWC has no first-task
effect in any case. Do not resume an old prompt-model checkpoint into this model.

## Continual learning after the overlap audit

`dgx_semantic.yaml` defines OEM then LandCover.ai, with EWC. Prepare OEM using
the analogous native export command and its sources.jsonl. OpenEarthMap includes
regions sourced from LandCover.ai: cross-dataset source overlap is NOT resolved
by independently splitting each dataset. Before running the combined experiment,
audit/remove shared source areas or assign all overlapping content to consistent
partitions across both tasks. `--exclude-regions name1 name2 ...` can remove
identified OEM source regions, using the first component of source_id. The report
deliberately sets `cross_dataset_overlap_audited: false`; exclusion alone does
not prove geographic disjointness. Reference attribution:
https://open-earth-map.org/attribution.html

Compare EWC strength 0 and nonzero, and scope all vs quantum. Validate strength;
100 is not guaranteed appropriate for the enlarged trainable network. Include
single-task baselines and final old-task forgetting. Current shared five-class
OEM labels differ from the original eight-class benchmark. This does not support
claiming an eight-class benchmark improvement.

## Verification and deployment

Unit tests cover decoder gradients, frozen SAM weights, quantum gate gradients,
rectangular outputs, trainable-state roundtrip, tiny-pattern overfitting, native
pixel preservation, and split preservation. A real random-weight SAM smoke checks
the full forward/backward path; it is a software check, not a quality result.

```bash
python -m pytest -q
python smoke_model.py --random-sam --decoder semantic
```

The upgrade is now tracked directly in the repository source files. Install
requirements-model.txt in the existing CUDA-capable environment. The ZIP files
in releases/ are historical snapshots; do not extract them over newer source.
The previous prompt-model configs are retained. Use a separate output directory.

## Existing prepared datasets (no preprocessing required)

Use configs/experiments/dgx_semantic_sequential_existing.yaml for the existing
OpenEarthMap -> LandCover.ai stream. It uses eight qubits, batch eight, two workers,
20 epochs per task, and one warmup epoch. Use
configs/experiments/dgx_semantic_landcover_existing.yaml for the separate
LandCover.ai-only diagnostic baseline. Native-resolution export above is optional;
existing resized inputs cannot recover detail lost during earlier resizing.

The sequential experiment uses EWC parameter anchors and Fisher statistics, with
no replay buffer, generated old-task examples, or old-task training batches.
Prior-task held-out data is used only for evaluation. Cross-dataset overlap still
requires an audit for a leakage-free research claim.

verify_learning.py runs a fixed training-subset diagnostic with separate validation
metrics. inspect_training_run.py reads saved settings and confusion statistics.
Neither synthetic tests nor training-subset scores establish benchmark accuracy.

Install requirements-report.txt to use results_report.py or full_results.ipynb.
Select the actual evaluation JSON from evaluate.py; the notebook's evaluation_final
path is an example and must match your run. Missing tasks are not fabricated.
