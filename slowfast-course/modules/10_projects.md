# Module 10 — Hands-On Projects

Completing these three projects will consolidate your understanding of the
entire PySlowFast pipeline.

---

## Project 1 — "What's That Action?" Custom Classifier

**Difficulty:** Beginner  
**Time:** 2–4 hours  
**Goal:** Fine-tune a pretrained SlowFast model on your own labelled video clips

### Overview

You will collect a tiny custom dataset (3–5 action classes), fine-tune a
pretrained checkpoint on it, and build a simple inference script.

### Step 1 — Collect Videos

Collect 20–30 short clips (5–10 seconds) for each of 3–5 action classes.
Ideas:
- Sports: `dribbling`, `passing`, `shooting`
- Office: `typing`, `writing`, `drinking_coffee`
- Exercise: `jumping_jacks`, `push_up`, `sit_up`

You can record with your phone or download from YouTube using `yt-dlp`.

Organise as:
```
my_dataset/
├── videos/
│   ├── train/
│   │   ├── typing/
│   │   │   ├── clip_001.mp4
│   │   │   └── ...
│   │   └── ...
│   └── val/
│       └── ...
```

### Step 2 — Generate CSV Files

```python
# scripts/project1_make_csv.py
import pathlib

CLASSES = ["typing", "writing", "drinking_coffee"]
ROOT    = pathlib.Path("my_dataset/videos")

for split in ["train", "val"]:
    rows = []
    for label, cls in enumerate(CLASSES):
        for video in (ROOT / split / cls).glob("*.mp4"):
            rows.append(f"{video.resolve()} {label}")
    with open(f"my_dataset/{split}.csv", "w") as f:
        f.write("\n".join(rows))
print("CSV files written.")
```

### Step 3 — Fine-Tune

```bash
# Download pretrained checkpoint
wget https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth \
  -O checkpoints/x3d_m.pyth

# Fine-tune
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_M.yaml \
  DATA.PATH_TO_DATA_DIR my_dataset \
  MODEL.NUM_CLASSES 3 \
  TRAIN.CHECKPOINT_FILE_PATH ./checkpoints/x3d_m.pyth \
  SOLVER.BASE_LR 0.005 \
  SOLVER.MAX_EPOCH 20 \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 4 \
  OUTPUT_DIR ./output/project1
```

### Step 4 — Evaluate and Demo

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_M.yaml \
  DATA.PATH_TO_DATA_DIR my_dataset \
  MODEL.NUM_CLASSES 3 \
  TEST.CHECKPOINT_FILE_PATH ./output/project1/checkpoints/checkpoint_best.pyth \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  NUM_GPUS 1
```

### Deliverables

- [ ] Train/val accuracy plot (TensorBoard screenshot)
- [ ] Confusion matrix for the 3 classes
- [ ] A short demo video showing the classifier in action
- [ ] Notes: what worked, what did not, what you would improve

---

## Project 2 — Architecture Ablation Study

**Difficulty:** Intermediate  
**Time:** 4–8 hours  
**Goal:** Compare C2D vs SlowFast on the same small dataset and understand the
accuracy/speed trade-off

### Overview

You will train three architectures on the same dataset (synthetic or your custom
one from Project 1) and compare:
- Training loss curves
- Final validation accuracy
- Inference speed (clips/second)
- Memory usage

### Step 1 — Train Three Models

```bash
# C2D baseline
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /tmp/synthetic_kinetics \
  MODEL.NUM_CLASSES 5 \
  NUM_GPUS 1 TRAIN.BATCH_SIZE 4 SOLVER.MAX_EPOCH 15 \
  TENSORBOARD.ENABLE True \
  OUTPUT_DIR ./output/ablation_c2d

# SlowFast R50
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_4x16_R50.yaml \
  DATA.PATH_TO_DATA_DIR /tmp/synthetic_kinetics \
  MODEL.NUM_CLASSES 5 \
  NUM_GPUS 1 TRAIN.BATCH_SIZE 4 SOLVER.MAX_EPOCH 15 \
  TENSORBOARD.ENABLE True \
  OUTPUT_DIR ./output/ablation_slowfast

# X3D-M
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_M.yaml \
  DATA.PATH_TO_DATA_DIR /tmp/synthetic_kinetics \
  MODEL.NUM_CLASSES 5 \
  NUM_GPUS 1 TRAIN.BATCH_SIZE 4 SOLVER.MAX_EPOCH 15 \
  TENSORBOARD.ENABLE True \
  OUTPUT_DIR ./output/ablation_x3d
```

### Step 2 — Measure Inference Speed

```python
# scripts/project2_benchmark.py
import time, torch
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model

def benchmark(cfg_file, checkpoint, n_runs=50):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.NUM_GPUS = 0
    cfg.freeze()

    model = build_model(cfg)
    model.eval()

    # Dummy input
    T  = cfg.DATA.NUM_FRAMES
    sp = cfg.DATA.TRAIN_CROP_SIZE
    dummy = [torch.zeros(1, 3, T, sp, sp)]  # single pathway simplification

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model(dummy)

    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    elapsed = time.time() - start

    clips_per_sec = n_runs / elapsed
    print(f"{cfg_file}: {clips_per_sec:.1f} clips/sec")

benchmark("configs/Kinetics/C2D_8x8_R50.yaml",        "")
benchmark("configs/Kinetics/SLOWFAST_4x16_R50.yaml",   "")
benchmark("configs/Kinetics/X3D_M.yaml",               "")
```

### Step 3 — Fill in the Comparison Table

| Model | Val Top-1 | Train Time | Inference Speed | GPU Memory |
|-------|-----------|-----------|-----------------|------------|
| C2D R50 | | | | |
| SlowFast R50 | | | | |
| X3D-M | | | | |

### Deliverables

- [ ] Completed comparison table
- [ ] TensorBoard overlaid loss curves for all three models
- [ ] Written analysis: when would you choose each architecture?

---

## Project 3 — Grad-CAM Interpretability Analysis

**Difficulty:** Intermediate to Advanced  
**Time:** 3–5 hours  
**Goal:** Use Grad-CAM to understand what spatial and temporal regions drive
predictions and evaluate model robustness

### Overview

You will run Grad-CAM on five test clips, analyse the attention maps, and
design an adversarial experiment to test whether the model is learning the
right features.

### Step 1 — Generate Grad-CAM Visualisations

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /data/kinetics400 \
  TEST.CHECKPOINT_FILE_PATH ./checkpoints/SLOWFAST_8x8_R50_K400.pkl \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TENSORBOARD.ENABLE True \
  TENSORBOARD.MODEL_VIS.ENABLE True \
  TENSORBOARD.MODEL_VIS.GRAD_CAM.ENABLE True \
  TENSORBOARD.MODEL_VIS.GRAD_CAM.LAYER_LIST \
    "['s5/pathway0_res2', 's5/pathway1_res2']" \
  OUTPUT_DIR ./output/gradcam
```

### Step 2 — Analysis Questions

For each Grad-CAM video examine:

1. Does the model attend to the **correct body part** for the action?
   (e.g., hands for "clapping", feet for "kicking")
2. Does the **Fast pathway** attend to different regions than the **Slow pathway**?
3. Are there background regions being lit up? (potential dataset bias)

### Step 3 — Adversarial Test

Create a version of your test clips where the **background is replaced** with a
static grey frame while the action remains the same. Does the model still
predict correctly?

```python
# scripts/project3_replace_background.py
import cv2, numpy as np

def replace_background(video_path: str, output_path: str, bg_color=(128, 128, 128)):
    """
    Naive background replacement: subtract the mean frame, fill with solid color.
    For a proper version use a segmentation model.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    frames = np.array(frames, dtype=np.float32)
    mean_frame = frames.mean(axis=0)

    # Simple diff-based foreground extraction
    diff = np.abs(frames - mean_frame).mean(axis=-1, keepdims=True)
    threshold = np.percentile(diff, 60)
    fg_mask = (diff > threshold).astype(np.float32)

    # Blend with solid background
    bg = np.full_like(frames, fill_value=bg_color)
    blended = frames * fg_mask + bg * (1 - fg_mask)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
    for frame in blended.astype(np.uint8):
        writer.write(frame)
    writer.release()
    print("Saved to", output_path)

replace_background("test_clip.mp4", "test_clip_nobg.mp4")
```

### Step 4 — Record Results

| Clip | Original Prediction | Without Background | Conclusion |
|------|--------------------|--------------------|-----------|
| clip_1.mp4 | | | |
| clip_2.mp4 | | | |
| ... | | | |

### Deliverables

- [ ] Grid of Grad-CAM frames for 5 test clips
- [ ] Written analysis of what each pathway attends to
- [ ] Table comparing predictions with/without background
- [ ] 200-word reflection: what does this tell you about the model's biases?

---

## Final Assessment

Complete all three projects and answer these questions:

1. Explain the SlowFast design in your own words to someone who knows basic
   machine learning but nothing about video models.
2. You need to deploy an action recognition system in a hospital to detect
   patient falls in real time. Which model would you use? How would you fine-tune
   it? What concerns about bias and reliability would you address?
3. What limitation of PySlowFast would make you look for another framework?

---

## Course Complete!

You have now covered:

- [x] Video understanding fundamentals and benchmarks
- [x] SlowFast dual-pathway architecture and its variants
- [x] Full environment setup from scratch
- [x] Dataset preparation for 4 major benchmarks
- [x] YAML config system with all key parameters
- [x] Single-GPU and multi-GPU training
- [x] Model evaluation with proper metrics
- [x] TensorBoard visualization, Grad-CAM, and confusion matrices
- [x] Running demos on real video and webcam
- [x] Three hands-on projects

### Further Reading

| Resource | Link |
|----------|------|
| SlowFast paper | https://arxiv.org/abs/1812.03982 |
| X3D paper | https://arxiv.org/abs/2004.04730 |
| MViT paper | https://arxiv.org/abs/2104.11227 |
| PyTorchVideo | https://pytorchvideo.org/ |
| Video Swin Transformer | https://arxiv.org/abs/2106.13230 |
| Kinetics dataset paper | https://arxiv.org/abs/1705.06950 |
| AVA dataset paper | https://arxiv.org/abs/1705.08421 |
