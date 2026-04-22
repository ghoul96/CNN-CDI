# Module 05 — The Config System

## 5.1 Why YAML Configs?

PySlowFast stores every hyperparameter in a YAML config file under `configs/`.
This ensures:

- **Reproducibility** — save the config alongside your checkpoints
- **Flexibility** — override any value on the command line without editing files
- **Discoverability** — all knobs are documented in `slowfast/config/defaults.py`

---

## 5.2 Config Hierarchy

```
slowfast/config/defaults.py        ← Base defaults for every parameter
           │
           ▼
configs/Kinetics/C2D_8x8_R50.yaml  ← Model-specific overrides
           │
           ▼
Command-line overrides              ← Highest priority
  e.g.  NUM_GPUS 2  TRAIN.BATCH_SIZE 16
```

Any parameter set at a lower level is overridden by the one above.

---

## 5.3 Anatomy of a Config File

Let us walk through `configs/Kinetics/SLOWFAST_8x8_R50.yaml` line by line:

```yaml
# ─── Training ─────────────────────────────────────────────────────────────
TRAIN:
  ENABLE: True          # Run training phase
  DATASET: kinetics     # Which dataset loader to use
  BATCH_SIZE: 64        # Total batch size across all GPUs
  EVAL_PERIOD: 10       # Evaluate every N epochs
  CHECKPOINT_PERIOD: 1  # Save checkpoint every N epochs
  AUTO_RESUME: True     # Resume from last checkpoint if it exists

# ─── Testing ──────────────────────────────────────────────────────────────
TEST:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
  NUM_ENSEMBLE_VIEWS: 10  # Number of temporal × spatial crops at test time
  NUM_SPATIAL_CROPS: 3

# ─── Data ─────────────────────────────────────────────────────────────────
DATA:
  PATH_TO_DATA_DIR: /path/to/kinetics400   # Folder with train.csv / val.csv
  NUM_FRAMES: 8          # T — frames for the Slow pathway
  SAMPLING_RATE: 8       # τ — stride between sampled frames
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3, 3]   # [Slow channels, Fast channels]

# ─── SlowFast specific ────────────────────────────────────────────────────
SLOWFAST:
  ALPHA: 8              # α — temporal ratio Fast/Slow
  BETA_INV: 8           # 1/β — channel reduction factor for Fast pathway
  FUSION_CONV_CHANNEL_RATIO: 2
  FUSION_KERNEL_SZ: 7

# ─── Model ────────────────────────────────────────────────────────────────
MODEL:
  NUM_CLASSES: 400      # Output classes (400 for K400)
  ARCH: slowfast        # Architecture name
  MODEL_NAME: SlowFast
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5

# ─── ResNet backbone ──────────────────────────────────────────────────────
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50             # ResNet-50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3, 3], [4, 4], [6, 6], [3, 3]]

# ─── Solver (optimisation) ────────────────────────────────────────────────
SOLVER:
  BASE_LR: 0.1          # Learning rate (scaled with batch size)
  LR_POLICY: cosine     # cosine annealing schedule
  MAX_EPOCH: 196
  MOMENTUM: 0.9
  WEIGHT_DECAY: 0.0001
  WARMUP_EPOCHS: 34.0
  WARMUP_START_LR: 0.01
  OPTIMIZING_METHOD: sgd

# ─── Hardware ─────────────────────────────────────────────────────────────
NUM_GPUS: 8
NUM_SHARDS: 1           # Number of machines (1 = single machine)
SHARD_ID: 0

# ─── Output ───────────────────────────────────────────────────────────────
OUTPUT_DIR: /tmp/slowfast_experiment
```

---

## 5.4 Key Parameters Explained

### Data Parameters

| Parameter | Meaning | Effect of increasing |
|-----------|---------|---------------------|
| `DATA.NUM_FRAMES` | T (Slow pathway frames) | More temporal context, slower |
| `DATA.SAMPLING_RATE` | τ (stride between frames) | Wider temporal window |
| `SLOWFAST.ALPHA` | α (Fast fps ratio) | More frames in Fast, more compute |
| `SLOWFAST.BETA_INV` | 1/β (Fast channel reduction) | More Fast capacity, less efficiency |
| `DATA.TRAIN_CROP_SIZE` | Spatial crop during training | Larger = more detail, more memory |

### Solver Parameters

| Parameter | Meaning | Common values |
|-----------|---------|---------------|
| `SOLVER.BASE_LR` | Learning rate for full batch | 0.1 (scale linearly with batch) |
| `SOLVER.LR_POLICY` | Schedule type | `cosine`, `steps_with_relative_lrs` |
| `SOLVER.MAX_EPOCH` | Total training epochs | 196–256 for K400 |
| `SOLVER.WARMUP_EPOCHS` | Gradual LR warmup | 34 for large-batch training |

**Linear scaling rule:** When you halve the number of GPUs, halve `BASE_LR` too:

```
BASE_LR = 0.1 × (total_batch / 256)   ← widely used heuristic
```

### Test Parameters

| Parameter | Meaning | Standard values |
|-----------|---------|-----------------|
| `TEST.NUM_ENSEMBLE_VIEWS` | Temporal clips per video | 10 |
| `TEST.NUM_SPATIAL_CROPS` | Spatial crops per clip | 3 |

Total predictions per video = `NUM_ENSEMBLE_VIEWS × NUM_SPATIAL_CROPS` = 30,
averaged via softmax.

---

## 5.5 Creating Your Own Config

Start from the closest existing config and override only what you need:

```yaml
# configs/MyExperiment/slowfast_5classes.yaml

# Inherit by explicitly referencing the base (convention, not a real feature)
# Copy configs/Kinetics/SLOWFAST_8x8_R50.yaml and change:

TRAIN:
  BATCH_SIZE: 8          # Smaller for a single GPU

DATA:
  PATH_TO_DATA_DIR: /tmp/synthetic_kinetics
  NUM_FRAMES: 4          # Smaller clip for faster iteration

MODEL:
  NUM_CLASSES: 5         # Our synthetic dataset has 5 classes

SOLVER:
  BASE_LR: 0.00125       # 0.1 × (8/64) for our smaller batch
  MAX_EPOCH: 20          # Quick experiment

NUM_GPUS: 1

OUTPUT_DIR: ./output/my_experiment
```

---

## 5.6 Command-Line Overrides

Any YAML key can be overridden via the command line as positional arguments:

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /my/data \     # override data path
  NUM_GPUS 2 \                          # override GPU count
  SOLVER.BASE_LR 0.05 \                 # override learning rate
  OUTPUT_DIR ./runs/exp_01              # override output directory
```

This is the preferred way to run hyperparameter sweeps — keep configs clean and
annotate differences only in the command.

---

## 5.7 Useful `defaults.py` Sections to Know

Open `slowfast/config/defaults.py` and look for these sections:

- `_C.DATA` — all data loading parameters
- `_C.SLOWFAST` — slow/fast pathway parameters
- `_C.MODEL` — model structure
- `_C.SOLVER` — learning rate, schedule, optimizer
- `_C.TENSORBOARD` — visualization options
- `_C.DEMO` — demo / inference options
- `_C.DETECTION` — AVA detection settings
- `_C.BN` — batch normalization settings

---

## 5.8 Knowledge Check

1. What is the priority order when a parameter is set in `defaults.py`, a YAML
   config, and the command line simultaneously?
2. If you train with 8 GPUs and `BASE_LR=0.1`, what should `BASE_LR` be for
   1-GPU training with the same per-GPU batch size?
3. What does `SLOWFAST.BETA_INV: 8` mean about the Fast pathway's channel count?
4. How would you disable training and only run testing from the command line?

---

## Next Module

[Module 06 — Training Your First Model →](06_training.md)
