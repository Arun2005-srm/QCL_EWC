# Train OpenEarthMap now, append LandCover.ai later

Leave LandCover.ai preprocessing running in its existing SSH session. Use a
second session for training. Preprocessing and the quantum simulator use CPU;
SAM training uses the selected CUDA GPU. CPU, host RAM and storage contention
can still slow both jobs. No DGX job is launched from this local workspace.

Extract this training update in `/raid/workspace/QCL_EWC/QCL_EWC`:

```bash
cd /raid/workspace/QCL_EWC/QCL_EWC
unzip -o staged_training.zip
source .venv/bin/activate
```

Set the actual existing pretrained SAM ViT-B checkpoint path. This is not the
alignment `task_00.pt` checkpoint and must not be a random-weight smoke artifact.

```bash
export SAM_CHECKPOINT=/absolute/path/to/sam_vit_b_01ec64.pth
test -f "$SAM_CHECKPOINT" || echo "Set SAM_CHECKPOINT to your downloaded ViT-B weights"
```

First verify the prepared OpenEarthMap data (this reads the images and can take
time), then train it alone:

```bash
python validate_data.py --config /raid/workspace/RS_Dataset/configs/openearthmap.yaml

python -u train.py --config configs/experiments/dgx_openearthmap.yaml \
  --sam-checkpoint "$SAM_CHECKPOINT" --device cuda \
  --output outputs/dgx_openearthmap
```

The runner currently reports progress once per epoch. It also needs time for
Fisher estimation and evaluation after the last epoch. Wait for task completion:
`outputs/dgx_openearthmap/task_00.pt` is written only after those stages. It
contains trained alignment/class prompts plus the consolidated EWC state.
Do not alter OpenEarthMap's config, manifest or data after training starts.
This uses the custom scene-level partitions you generated, not the original
OpenEarthMap official split assignments.

After BOTH OpenEarthMap training and LandCover.ai preprocessing finish:

```bash
python validate_data.py --config /raid/workspace/RS_Dataset/configs/landcover_ai.yaml

python -u train.py --config configs/experiments/dgx_prepared.yaml \
  --sam-checkpoint "$SAM_CHECKPOINT" --device cuda \
  --continue-from outputs/dgx_openearthmap/task_00.pt \
  --output outputs/dgx_oem_then_landcover
```

The combined config lists OpenEarthMap then LandCover.ai, but the continuation
starts at the new LandCover.ai task. OpenEarthMap is not retrained. Its saved
Fisher and parameter anchors constrain task 2, and both test sets are evaluated
after task 2 to measure retention. New task optimization resets Adam, just as
the uninterrupted training loop does.

`--continue-from` is different from `--resume`: continuation allows appended
datasets only after the previous run is fully complete. It checks that the old
dataset configurations, discovered partitions, SAM weights, model, training,
loader and EWC settings match. It rejects mid-task checkpoints or protocol
changes. Use a new output directory for the appended run.

If a stage is interrupted, resume that same stage and config using its `last.pt`:

```bash
python -u train.py --config configs/experiments/dgx_openearthmap.yaml \
  --sam-checkpoint "$SAM_CHECKPOINT" --device cuda \
  --resume outputs/dgx_openearthmap/last.pt --output outputs/dgx_openearthmap
```

Keep long-running sessions connected, or run inside your existing terminal
multiplexer/job scheduler. No monitoring or automatic stage-2 launch is set up.
