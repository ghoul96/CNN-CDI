# Module 06 — Training Your First Model

## 6.1 The Single Entry-Point

Every operation in PySlowFast — training, validation, testing, and demo — goes
through ONE script:

```
python tools/run_net.py --cfg <config_file> [KEY VALUE ...]
```

The script reads the config, builds the model, data loaders, optimizer, and
orchestrates the training loop. You never need to edit `run_net.py`.

---

## 6.2 Quickstart: Single-GPU Training on the Synthetic Dataset

First, generate the synthetic dataset (see Module 04, Section 4.6).  
Then run:

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /tmp/synthetic_kinetics \
  MODEL.NUM_CLASSES 5 \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 4 \
  DATA.NUM_FRAMES 8 \
  SOLVER.BASE_LR 0.00125 \
  SOLVER.MAX_EPOCH 10 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  TEST.ENABLE True \
  OUTPUT_DIR ./output/c2d_synthetic
```

What to watch in the terminal:
```
Epoch: [1][  1/  5]  Loss: 1.61  Lr: 0.000250  Top1: 20.0  Top5: 100.0
Epoch: [1][  5/  5]  Loss: 1.55  Lr: 0.001250  Top1: 22.0  Top5: 100.0
...
```

- **Loss should decrease** over epochs
- **Top1** (accuracy) should rise above random (20% for 5 classes)

---

## 6.3 Training C2D → SLOWFAST: Upgrading Your Config

| Config | Architecture | GPU Memory | Notes |
|--------|-------------|-----------|-------|
| `C2D_8x8_R50.yaml` | C2D R50 | ~6 GB | Simplest, good starting point |
| `SLOW_4x16_R50.yaml` | Slow-only R50 | ~8 GB | Intermediate complexity |
| `SLOWFAST_4x16_R50.yaml` | SlowFast R50 | ~10 GB | Full SlowFast baseline |
| `SLOWFAST_8x8_R50.yaml` | SlowFast R50 | ~16 GB | Best SlowFast baseline |
| `X3D_M.yaml` | X3D-M | ~5 GB | Efficient, great for mobile |
| `MVIT_B_16x4.yaml` | MViTv1 Base | ~18 GB | Transformer architecture |

Switch architecture by simply changing the `--cfg` argument.

---

## 6.4 Multi-GPU Training (Data Parallel)

PySlowFast uses PyTorch `DistributedDataParallel` (DDP) automatically when
`NUM_GPUS > 1`:

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /data/kinetics400 \
  NUM_GPUS 4 \
  TRAIN.BATCH_SIZE 32 \       # 8 per GPU
  SOLVER.BASE_LR 0.05 \       # 0.1 × (32/64)
  OUTPUT_DIR ./output/slowfast_k400
```

> **Rule of thumb:** total batch size = `NUM_GPUS × (BATCH_SIZE / NUM_GPUS)`.
> Keep per-GPU batch size at 8 for stability.

---

## 6.5 Multigrid Training (3–6× Speedup)

Multigrid training varies the spatial and temporal resolution of input clips
cyclically during training. The model sees many more clips per second while
maintaining final accuracy.

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50_stepwise_multigrid.yaml \
  DATA.PATH_TO_DATA_DIR /data/kinetics400 \
  NUM_GPUS 8 \
  OUTPUT_DIR ./output/slowfast_multigrid
```

Multigrid-trained SlowFast R50 achieves 76.6% on K400 vs 76.8% with standard
training but trains **4× faster**.

---

## 6.6 Fine-Tuning from a Pretrained Checkpoint

Instead of training from scratch (which requires days on K400), download a
pretrained model from the Model Zoo and fine-tune on your own data:

```bash
# 1. Download pretrained checkpoint
#    Find links in MODEL_ZOO.md:
#    e.g. SlowFast R50 K400:
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl \
  -O ./checkpoints/SLOWFAST_8x8_R50_K400.pkl

# 2. Fine-tune on your dataset (e.g., 10 custom classes)
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /data/my_custom_dataset \
  MODEL.NUM_CLASSES 10 \
  TRAIN.CHECKPOINT_FILE_PATH ./checkpoints/SLOWFAST_8x8_R50_K400.pkl \
  TRAIN.CHECKPOINT_TYPE pytorch \
  SOLVER.BASE_LR 0.01 \
  SOLVER.MAX_EPOCH 30 \
  NUM_GPUS 1 \
  OUTPUT_DIR ./output/finetuned_custom
```

> Lower LR (`0.01` vs `0.1`) when fine-tuning to avoid destroying pre-trained
> features.

---

## 6.7 Resuming an Interrupted Training Run

If training is interrupted (power loss, OOM, time limit), just re-run the
exact same command. With `TRAIN.AUTO_RESUME: True` (default), PySlowFast
detects the last checkpoint and continues:

```
Loading checkpoint from ./output/slowfast_k400/checkpoints/checkpoint_epoch_00025.pyth
```

Or point explicitly to a checkpoint:

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  TRAIN.CHECKPOINT_FILE_PATH ./output/slowfast_k400/checkpoints/checkpoint_epoch_00025.pyth \
  TRAIN.CHECKPOINT_TYPE pytorch \
  OUTPUT_DIR ./output/slowfast_k400
```

---

## 6.8 Understanding the Output Directory

After training, your `OUTPUT_DIR` will contain:

```
output/c2d_synthetic/
├── checkpoints/
│   ├── checkpoint_epoch_00001.pyth
│   ├── checkpoint_epoch_00010.pyth   ← last epoch
│   └── checkpoint_best.pyth          ← best val accuracy
├── logs/
│   └── log.txt                       ← full training log
└── runs-kinetics/                    ← TensorBoard events (if enabled)
```

---

## 6.9 Monitoring Training

```bash
# Read the log directly
tail -f ./output/c2d_synthetic/logs/log.txt

# Or launch TensorBoard (requires TENSORBOARD.ENABLE: True in config)
tensorboard --logdir ./output/c2d_synthetic/runs-kinetics
# Open http://localhost:6006
```

---

## 6.10 Common Training Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| OOM (out of memory) | `RuntimeError: CUDA out of memory` | Halve `TRAIN.BATCH_SIZE`, or use `DATA.NUM_FRAMES` 4 |
| Loss NaN | Loss goes to `nan` after a few steps | Lower `SOLVER.BASE_LR` by 10× |
| Slow data loading | GPU utilisation < 30% | Increase `DATA_LOADER.NUM_WORKERS` (try 8–16) |
| Poor accuracy | Stuck at random chance | Check CSV label format, NUM_CLASSES, LR scale |
| Checkpoint not found | `FileNotFoundError` | Verify `CHECKPOINT_FILE_PATH` is correct |

---

## 6.11 Knowledge Check

1. What single flag disables training so only testing runs?
2. You have 2 GPUs instead of 8. Starting from `BASE_LR=0.1` with the original
   8-GPU batch size of 64, what `BASE_LR` should you use if you keep per-GPU
   batch size the same?
3. What is the benefit of Multigrid training vs standard training?
4. Name two things you must change when fine-tuning a K400 pretrained model on a
   custom 10-class dataset.

---

## Next Module

[Module 07 — Evaluation & Testing →](07_evaluation.md)
