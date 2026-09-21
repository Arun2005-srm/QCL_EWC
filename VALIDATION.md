# Local validation

Executed on Windows, Python 3.10, CPU, using PyTorch 2.14.0+cpu,
torchvision 0.29.0, PennyLane 0.42.3, NumPy 2.2.6, Pillow 12.3.0,
PyYAML 6.0.3, tifffile 2025.5.10, and pytest 9.1.1.
Official SAM source revision: `dca509fe793f601edb92606367a655c15ac00fdf`.

Verified:

- Dataset tests: pairing, unknown labels, indexed/palette/RGB/binary masks,
  multiband TIFFs, scene-disjoint splits, split-manifest round trips,
  inherited paths, tensor dtypes/shapes, ignored edge padding, paired flips.
- Data smoke: nine generated scenes, 36 tiles, one segmentation training step,
  held-out loaders; passed with zero and two worker processes.
- All supplied dataset YAMLs and the ordered two-dataset configuration parse.
- `validate_data.py` writes reports, resolved configurations, and split manifests.
- EWC tests: nonnegative Fisher, zero anchor penalty, positive perturbed penalty,
  individual-gradient squaring, state round trip, all-ignored loss handling.
- Two-task toy-model training and task-boundary resume produce identical final
  weights and evaluation matrices.
- Real PennyLane circuit: gradients through both projection layers and circuit
  angles; state-dictionary round trip.
- Actual SAM ViT-B architecture, **random weights**, five-class forward/backward:
  all trainable parameter groups have finite nonzero gradients; SAM stays frozen.
- Actual SAM ViT-B + quantum alignment + EWC, **random weights and synthetic
  data**: two tasks complete training, validation, Fisher estimation, evaluation
  of seen tasks, and checkpoint saving. Approximately 77 seconds on this CPU.
- Dependency consistency check and Python compilation pass.

Local machine-generated reports are in ignored `outputs/`, including
`sam_two_task_smoke.json`, `smoke_stream/metrics.json`, and `worker_smoke.json`.
Tests with optional dependencies can be reproduced using the README commands.

Not established by these checks: real OpenEarthMap/LandCover.ai training,
pretrained SAM integration with a supplied checkpoint, DGX/CUDA execution,
segmentation accuracy, reduced forgetting, or quantum advantage. The repository
includes explicit commands for those next experiments; no pretrained checkpoint
or remote-sensing dataset was available locally.
