# Module 08 — Visualization Tools

## 8.1 Overview

PySlowFast ships three categories of visualization tools:

| Category | What you see | When to use |
|----------|-------------|-------------|
| **Training metrics** | Loss, accuracy, confusion matrix, histograms | During and after training |
| **Model analysis** | Weights, feature maps, Grad-CAM | Debugging, interpretability |
| **Demo / Inference** | Labelled bounding boxes on real video | Showing results to stakeholders |

All tools share the same TensorBoard infrastructure. Enable them by adding
`TENSORBOARD.ENABLE: True` to your config.

---

## 8.2 Enabling TensorBoard

Add to your YAML config:

```yaml
TENSORBOARD:
  ENABLE: True
  LOG_DIR: ""          # Leave empty → uses OUTPUT_DIR/runs-{DATASET}
  CLASS_NAMES_PATH: ""  # Optional: path to Kinetics/AVA class-name JSON
```

Then launch TensorBoard:
```bash
tensorboard --logdir ./output/my_experiment/runs-kinetics
# Navigate to http://localhost:6006
```

---

## 8.3 Training Metrics

### 8.3.1 Loss and Accuracy Curves

You get one curve per phase:
- `Train/loss`, `Train/top1_acc`, `Train/top5_acc`
- `Val/loss`, `Val/top1_acc`, `Val/top5_acc`

What to look for:
- Training loss steadily decreases → model is learning
- Val loss decreases alongside train loss → good generalisation
- Train loss continues decreasing but val loss increases → **overfitting**

### 8.3.2 Confusion Matrix

Enable with:
```yaml
TENSORBOARD:
  ENABLE: True
  CLASS_NAMES_PATH: /path/to/kinetics_classnames.json
  CONFUSION_MATRIX:
    ENABLE: True
    SUBSET_PATH: /path/to/subset_classes.txt  # Optional: visualise a subset
```

The confusion matrix shows:
- Rows = true class
- Columns = predicted class
- Diagonal = correctly classified

Dark off-diagonal cells indicate which classes are being confused with each other.

### 8.3.3 Prediction Histograms

Shows the top-k most frequently predicted classes per true class:

```yaml
TENSORBOARD:
  HISTOGRAM:
    ENABLE: True
    TOP_K: 10
    SUBSET_PATH: /path/to/subset.txt
```

Download the official class-name files:
```bash
# Kinetics
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json

# AVA
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/ava_classids.json

# Kinetics parent-child mapping (for hierarchical confusion matrix)
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/parents.json
```

---

## 8.4 Model Analysis

### 8.4.1 Enable Model Visualization

```yaml
TENSORBOARD:
  ENABLE: True
  MODEL_VIS:
    ENABLE: True
    MODEL_WEIGHTS: True    # Visualise weight histograms
    ACTIVATIONS: True      # Visualise feature map activations
    INPUT_VIDEO: True      # Show input video alongside feature maps
    LAYER_LIST:            # Which layers to inspect
      - "s1/pathway0_stem/conv"
      - "s5/pathway0_res2"
      - "s5/pathway1_res2"
```

### 8.4.2 Weights Visualization

Displays histograms of weight distributions per layer over training.

Use case:
- Check for dead neurons (weights collapsed to 0)
- Check for exploding weights

### 8.4.3 Feature Maps

Shows the activation maps at specified layers as the video plays.
This reveals what spatial regions each layer is "looking at".

### 8.4.4 Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM highlights the video regions that most influenced the prediction.
Requires specifying one layer per pathway:

```yaml
TENSORBOARD:
  MODEL_VIS:
    GRAD_CAM:
      ENABLE: True
      LAYER_LIST:
        - "s5/pathway0_res2"   # Slow pathway final residual block
        - "s5/pathway1_res2"   # Fast pathway final residual block
```

The output is a heatmap overlaid on the original video frames.

- **Red/warm regions** → strongly activated by the predicted class
- **Blue/cool regions** → not contributing to the prediction

---

## 8.5 Running the Demo on a Video File

The demo runs a trained model on any MP4 file and writes an annotated output:

```yaml
# Add to your config:
DEMO:
  ENABLE: True
  LABEL_FILE_PATH: /path/to/kinetics_classnames.json
  INPUT_VIDEO: /path/to/input.mp4
  OUTPUT_FILE: /path/to/output.mp4
  THREAD_ENABLE: True          # Use background thread for reader/writer
  NUM_VIS_INSTANCES: 2         # CPU processes for rendering
  NUM_CLIPS_SKIP: 0            # Skip N clips between predictions
  DISPLAY_WIDTH: 0             # 0 = use input video width
  DISPLAY_HEIGHT: 0
```

Run:
```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  TEST.CHECKPOINT_FILE_PATH ./checkpoints/SLOWFAST_8x8_R50_K400.pkl \
  TRAIN.ENABLE False \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO ./sample_video.mp4 \
  DEMO.OUTPUT_FILE ./demo_output.mp4 \
  DEMO.LABEL_FILE_PATH ./kinetics_classnames.json
```

---

## 8.6 Webcam Real-Time Demo

Replace `DEMO.INPUT_VIDEO` with `DEMO.WEBCAM`:

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_M.yaml \
  TEST.CHECKPOINT_FILE_PATH ./checkpoints/x3d_m.pyth \
  TRAIN.ENABLE False \
  DEMO.ENABLE True \
  DEMO.WEBCAM 0 \              # 0 = default webcam
  DEMO.LABEL_FILE_PATH ./kinetics_classnames.json \
  DEMO.NUM_CLIPS_SKIP 1        # Skip 1 clip for smoother display
```

> X3D-M is recommended for real-time use due to its low computation cost.

---

## 8.7 AVA Action Detection Demo

For detection models you can overlay bounding boxes:

```yaml
DEMO:
  ENABLE: True
  OUTPUT_FILE: ./ava_demo_output.mp4
  LABEL_FILE_PATH: /path/to/ava_classids.json
  INPUT_VIDEO: /path/to/ava_frames/video_name   # Frame folder, not MP4
  PREDS_BOXES: /path/to/predicted_boxes.csv     # Pre-computed person detections
  GT_BOXES: /path/to/ava_val_v2.2.csv           # Optional ground truth
```

---

## 8.8 Practical Exercise: Visualize C2D on Your Synthetic Dataset

```bash
# 1. Train C2D for 10 epochs with TensorBoard enabled
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /tmp/synthetic_kinetics \
  MODEL.NUM_CLASSES 5 \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 4 \
  SOLVER.MAX_EPOCH 10 \
  TENSORBOARD.ENABLE True \
  TENSORBOARD.MODEL_VIS.ENABLE True \
  TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE True \
  OUTPUT_DIR ./output/c2d_vis

# 2. Launch TensorBoard while training
tensorboard --logdir ./output/c2d_vis

# 3. After training, run demo
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  TEST.CHECKPOINT_FILE_PATH ./output/c2d_vis/checkpoints/checkpoint_best.pyth \
  TRAIN.ENABLE False \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO /tmp/synthetic_kinetics/videos/val/typing/0000.mp4 \
  DEMO.OUTPUT_FILE ./demo_typing.mp4 \
  DEMO.LABEL_FILE_PATH ./my_classnames.json \
  MODEL.NUM_CLASSES 5
```

---

## 8.9 Knowledge Check

1. What does a dark off-diagonal cell in a confusion matrix indicate?
2. Grad-CAM requires one entry in `LAYER_LIST` per pathway. Why?
3. Why is X3D-M recommended for webcam demos over SlowFast R50?
4. What does `DEMO.NUM_CLIPS_SKIP` do, and when would you increase it?

---

## Next Module

[Module 09 — Running the Full Demo →](09_demo.md)
