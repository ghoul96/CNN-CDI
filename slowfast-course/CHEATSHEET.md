# PySlowFast Quick-Reference Cheat Sheet

## Installation (one-liner summary)

```bash
conda create -n slowfast python=3.9 -y && conda activate slowfast
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install 'git+https://github.com/facebookresearch/fvcore' iopath simplejson \
            psutil opencv-python tensorboard pytorchvideo moviepy \
            'git+https://github.com/facebookresearch/fairscale'
conda install av -c conda-forge -y
git clone https://github.com/facebookresearch/slowfast && cd slowfast
python setup.py build develop
export PYTHONPATH=$(pwd)/slowfast:$PYTHONPATH
```

---

## Command Patterns

| Task | Command |
|------|---------|
| Train from scratch | `python tools/run_net.py --cfg <cfg>` |
| Train + test | Add `TEST.ENABLE True` |
| Test only | Add `TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH <ckpt>` |
| Fine-tune | Add `TRAIN.CHECKPOINT_FILE_PATH <pretrained>` |
| Demo on video | Add `DEMO.ENABLE True DEMO.INPUT_VIDEO <vid>` |
| Demo on webcam | Add `DEMO.ENABLE True DEMO.WEBCAM 0` |
| CPU mode | Add `NUM_GPUS 0` |
| Override any param | `KEY VALUE KEY2 VALUE2` after `--cfg` |

---

## Key Config Parameters

```
NUM_GPUS                   Number of GPUs
TRAIN.BATCH_SIZE           Total batch across all GPUs
TRAIN.CHECKPOINT_FILE_PATH Path to resume / pretrained checkpoint
TEST.CHECKPOINT_FILE_PATH  Path to eval checkpoint
DATA.PATH_TO_DATA_DIR      Folder containing train.csv / val.csv
DATA.NUM_FRAMES            T  — slow pathway frames
DATA.SAMPLING_RATE         τ  — frame stride
SLOWFAST.ALPHA            α  — fast/slow fps ratio
SLOWFAST.BETA_INV         1/β — channel reduction for fast pathway
MODEL.NUM_CLASSES          Output classes
SOLVER.BASE_LR             Learning rate  (scale: 0.1 × batch / 256)
SOLVER.MAX_EPOCH           Training epochs
OUTPUT_DIR                 Where to save checkpoints + logs
TENSORBOARD.ENABLE         True/False — enable TensorBoard logging
DEMO.ENABLE                True/False — run inference demo
DEMO.INPUT_VIDEO           Path to input video file
DEMO.OUTPUT_FILE           Path to write output video
DEMO.LABEL_FILE_PATH       JSON mapping class name → id
```

---

## Architectures at a Glance

| ARCH flag | Config prefix | Memory | Speed | Notes |
|-----------|--------------|--------|-------|-------|
| `c2d` | C2D_8x8_R50 | Low | Fast | Baseline, 2-D conv + pool |
| `i3d` | I3D_8x8_R50 | High | Slow | Inflated 3-D, good ImageNet init |
| `slow` | SLOW_8x8_R50 | Med | Med | Slow pathway only |
| `slowfast` | SLOWFAST_8x8_R50 | Med | Med | Dual pathway flagship |
| `x3d` | X3D_M | Low | Fast | Best efficiency |
| `mvit` | MVIT_B_16x4 | Med | Slow | Transformer, SOTA accuracy |

---

## Pretrained Checkpoints (K400)

```
X3D-M          → https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth
C2D R50        → .../model_zoo/kinetics400/C2D_NOPOOL_8x8_R50.pkl
I3D R50        → .../model_zoo/kinetics400/I3D_8x8_R50.pkl
SlowFast R50   → .../model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl
MViT-B 16x4    → .../model_zoo/mvit/MVIT_B_16x4_CONV.pyth

base URL: https://dl.fbaipublicfiles.com/pyslowfast/
```

---

## Class Name Files

```
Kinetics:        https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json
AVA:             https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/ava_classids.json
Parent mapping:  https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/parents.json
```

---

## TensorBoard Visualization Flags

```yaml
TENSORBOARD:
  ENABLE: True
  CLASS_NAMES_PATH: kinetics_classnames.json
  CONFUSION_MATRIX:
    ENABLE: True
  HISTOGRAM:
    ENABLE: True
    TOP_K: 10
  MODEL_VIS:
    ENABLE: True
    MODEL_WEIGHTS: True
    ACTIVATIONS: True
    INPUT_VIDEO: True
    LAYER_LIST: ["s5/pathway0_res2", "s5/pathway1_res2"]
    GRAD_CAM:
      ENABLE: True
      LAYER_LIST: ["s5/pathway0_res2", "s5/pathway1_res2"]
```

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `No module named slowfast` | Set `PYTHONPATH=/path/to/slowfast/slowfast:$PYTHONPATH` |
| CUDA out of memory | Halve `TRAIN.BATCH_SIZE` or reduce `DATA.NUM_FRAMES` |
| Loss NaN | Reduce `SOLVER.BASE_LR` by 10× |
| `KeyError: model_state` | Add `TRAIN.CHECKPOINT_TYPE caffe2` for `.pkl` files |
| Slow data loading | Increase `DATA_LOADER.NUM_WORKERS` to 8–16 |
| Shape mismatch | Config and checkpoint must belong to the same model |
