# Module 07 — Evaluation & Testing

## 7.1 Evaluation vs Training

| Phase | Purpose | Flag |
|-------|---------|------|
| `TRAIN` | Update model weights | `TRAIN.ENABLE True` |
| `TEST` | Measure accuracy on held-out set | `TEST.ENABLE True` |
| `DEMO` | Run inference on arbitrary video | `DEMO.ENABLE True` |

You can run any combination. The most common patterns:

```bash
# Train and evaluate simultaneously (default)
TRAIN.ENABLE True   TEST.ENABLE True

# Evaluate a saved checkpoint only
TRAIN.ENABLE False  TEST.ENABLE True  TEST.CHECKPOINT_FILE_PATH path/to/ckpt.pyth
```

---

## 7.2 Run Testing Only

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /data/kinetics400 \
  TEST.CHECKPOINT_FILE_PATH ./checkpoints/SLOWFAST_8x8_R50_K400.pkl \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  NUM_GPUS 1
```

---

## 7.3 The Model Zoo — Pretrained Checkpoints

All pretrained models are listed in `MODEL_ZOO.md`. Below are key baselines:

### Kinetics-400 Action Classification

| Model | Backbone | Frames×Stride | Top-1 | Top-5 | Download |
|-------|----------|--------------|-------|-------|----------|
| C2D | R50 | 8×8 | 67.2 | 87.8 | [link](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/C2D_NOPOOL_8x8_R50.pkl) |
| I3D | R50 | 8×8 | 73.5 | 90.8 | [link](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/I3D_8x8_R50.pkl) |
| Slow | R50 | 8×8 | 74.8 | 91.6 | [link](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOW_8x8_R50.pkl) |
| SlowFast | R50 | 4×16 | 75.6 | 92.0 | [link](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl) |
| SlowFast | R50 | 8×8 | 77.0 | 92.6 | [link](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl) |
| X3D-M | — | 16×5 | 75.1 | 76.2 | [link](https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth) |
| MViTv1-B | Conv | 16×4 | 78.4 | 93.5 | [link](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvit/MVIT_B_16x4_CONV.pyth) |
| MViTv2-B | — | 32×3 | 82.9 | 95.7 | (see MODEL_ZOO.md) |

### AVA Action Detection (mAP)

| Model | Backbone | Pre-train | mAP |
|-------|----------|-----------|-----|
| SlowFast | R101 | K600 | 29.1 |
| SlowFast | R101 | K600 (16×8) | 29.4 |

---

## 7.4 Understanding Metrics

### Top-1 Accuracy
The fraction of videos where the predicted class with the **highest probability**
matches the ground truth:

$$\text{Top-1} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$$

### Top-5 Accuracy
The fraction where the ground truth is among the **5 highest** predicted classes.
Useful for datasets with many similar classes (e.g., Kinetics-400).

$$\text{Top-5} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[y_i \in \text{top-5}(\hat{\mathbf{p}}_i)]$$

### Mean Average Precision (mAP) — for AVA
For detection we compute AP per action class and average:

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c$$

where AP (area under the Precision-Recall curve) is computed per class `c`.

---

## 7.5 Test-Time Ensembling

At test time PySlowFast samples **multiple clips** from each video and averages
predictions. This consistently adds 1–3% accuracy:

```
NUM_ENSEMBLE_VIEWS = 10   → 10 different temporal windows
NUM_SPATIAL_CROPS  = 3    → left/centre/right crop for each window
─────────────────────────
Total clips per video = 30   (scores averaged)
```

To disable ensembling for faster evaluation:
```yaml
TEST:
  NUM_ENSEMBLE_VIEWS: 1
  NUM_SPATIAL_CROPS: 1
```

---

## 7.6 Checkpoint File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PyTorch | `.pyth` | Native PySlowFast format |
| Pickle | `.pkl` | Legacy Caffe2 format from original paper |

Specify the format to the loader:
```bash
TEST.CHECKPOINT_FILE_PATH ./model.pkl  TRAIN.CHECKPOINT_TYPE caffe2
TEST.CHECKPOINT_FILE_PATH ./model.pyth TRAIN.CHECKPOINT_TYPE pytorch   # default
```

---

## 7.7 Evaluating on Your Custom Dataset

After fine-tuning, evaluate on your held-out test split:

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /data/my_dataset \
  MODEL.NUM_CLASSES 10 \
  TEST.CHECKPOINT_FILE_PATH ./output/finetuned/checkpoints/checkpoint_best.pyth \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  NUM_GPUS 1
```

The output will print:
```
Testing with 1 GPUs
Test Epoch:  1/1  top1_acc: 87.40  top5_acc: 99.10
```

---

## 7.8 Saving Predictions to a File

To save per-video predictions for downstream analysis:

```yaml
TEST:
  SAVE_RESULTS_PATH: ./predictions.pkl
```

Then load in Python:
```python
import pickle

with open("./predictions.pkl", "rb") as f:
    results = pickle.load(f)
# results is a list of (video_id, predicted_label, [class_scores...])
```

---

## 7.9 Knowledge Check

1. What is the difference between Top-1 and Top-5 accuracy?
2. Why does using 30 clips per video during testing improve accuracy?
3. How would you load a pretrained checkpoint that was saved in Caffe2 format?
4. What metric is used for action detection (AVA) instead of Top-1 accuracy?

---

## Next Module

[Module 08 — Visualization Tools →](08_visualization.md)
