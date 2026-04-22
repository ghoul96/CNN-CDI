# Module 09 — Running the Full Demo

## 9.1 End-to-End Demo Walkthrough

This module guides you through running a complete action recognition demo from
scratch in under 30 minutes, using only a single GPU (or CPU for inference).

---

## 9.2 What You Need

- [ ] PySlowFast installed (Module 03)
- [ ] A pretrained checkpoint (download below)
- [ ] A short MP4 video file (any action video works)
- [ ] The Kinetics class names JSON file

---

## 9.3 Step-by-Step: Action Recognition on a Video

### Step 1 — Download a pretrained X3D-M checkpoint

X3D-M is the recommended starter model: only 4.73 GFLOPs and 3.8M parameters,
yet achieves 75.1% Top-1 on Kinetics-400.

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_m.pyth \
  -O checkpoints/x3d_m.pyth
```

### Step 2 — Download the Kinetics class names

```bash
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json
```

### Step 3 — Get a sample video

```bash
# Option A: download a short YouTube clip with yt-dlp
pip install yt-dlp
yt-dlp -x --audio-format mp3 -f "bestvideo[ext=mp4][height<=480]" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o sample_video.mp4

# Option B: use ffmpeg to generate a synthetic video
ffmpeg -f lavfi -i "testsrc=duration=5:size=320x240:rate=30" \
  -pix_fmt yuv420p sample_video.mp4
```

### Step 4 — Run the demo

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_M.yaml \
  TEST.CHECKPOINT_FILE_PATH ./checkpoints/x3d_m.pyth \
  TRAIN.ENABLE False \
  TEST.ENABLE False \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO ./sample_video.mp4 \
  DEMO.OUTPUT_FILE ./demo_output.mp4 \
  DEMO.LABEL_FILE_PATH ./kinetics_classnames.json \
  NUM_GPUS 1
```

### Step 5 — View the output

Open `demo_output.mp4`. Each frame is annotated with:
- The **predicted action class** name
- The **confidence score** (0–1)

---

## 9.4 CPU-Only Inference

If you do not have a GPU, set `NUM_GPUS 0`:

```bash
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_M.yaml \
  TEST.CHECKPOINT_FILE_PATH ./checkpoints/x3d_m.pyth \
  TRAIN.ENABLE False \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO ./sample_video.mp4 \
  DEMO.OUTPUT_FILE ./demo_cpu.mp4 \
  DEMO.LABEL_FILE_PATH ./kinetics_classnames.json \
  NUM_GPUS 0
```

Expect ~2–5 seconds per clip on CPU. Real-time is not possible without a GPU.

---

## 9.5 Using a Different Checkpoint and Config

To use SlowFast R50 instead:

```bash
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl \
  -O checkpoints/slowfast_r50.pkl

python tools/run_net.py \
  --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml \
  TEST.CHECKPOINT_FILE_PATH ./checkpoints/slowfast_r50.pkl \
  TRAIN.CHECKPOINT_TYPE caffe2 \         # .pkl = caffe2 format
  TRAIN.ENABLE False \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO ./sample_video.mp4 \
  DEMO.OUTPUT_FILE ./demo_slowfast.mp4 \
  DEMO.LABEL_FILE_PATH ./kinetics_classnames.json \
  NUM_GPUS 1
```

---

## 9.6 Building an Inference Script in Python

For integrating PySlowFast into your own application:

```python
# scripts/infer.py
"""
Minimal inference wrapper around PySlowFast.
Usage: python scripts/infer.py --video my_video.mp4 --checkpoint x3d_m.pyth
"""
import argparse
import json
import torch
import numpy as np
import cv2

# After installing PySlowFast and setting PYTHONPATH
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.datasets.utils import pack_pathway_output
import slowfast.utils.checkpoint as cu


def load_model(cfg_file: str, checkpoint_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_path
    cfg.NUM_GPUS = 1 if torch.cuda.is_available() else 0
    cfg.freeze()

    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    return model, cfg


def preprocess_clip(frames: np.ndarray, cfg) -> list:
    """
    frames: np.ndarray of shape [T, H, W, 3], uint8
    Returns a list of tensors (one per pathway).
    """
    # Resize to cfg.DATA.TEST_CROP_SIZE
    crop = cfg.DATA.TEST_CROP_SIZE
    resized = np.stack([cv2.resize(f, (crop, crop)) for f in frames])

    # Normalize ([0,255] → [0,1] → subtract mean, divide std)
    mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    std  = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    tensor = (resized.astype(np.float32) / 255.0 - mean) / std

    # Shape: [T, H, W, C] → [C, T, H, W] → [1, C, T, H, W]
    tensor = torch.from_numpy(tensor).permute(3, 0, 1, 2).unsqueeze(0)

    return pack_pathway_output(cfg, tensor)


def predict(model, inputs: list, class_names: list, top_k: int = 5):
    with torch.no_grad():
        preds = model(inputs)   # [1, num_classes]
    probs = torch.softmax(preds, dim=1)[0]
    top_probs, top_idxs = probs.topk(top_k)
    return [(class_names[i], p.item()) for i, p in zip(top_idxs, top_probs)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cfg",        default="configs/Kinetics/X3D_M.yaml")
    parser.add_argument("--classes",    default="kinetics_classnames.json")
    args = parser.parse_args()

    with open(args.classes) as f:
        class_names = list(json.load(f).keys())

    model, cfg = load_model(args.cfg, args.checkpoint)

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < cfg.DATA.NUM_FRAMES * cfg.SLOWFAST.ALPHA:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        print("Could not read frames from video")
        raise SystemExit(1)

    inputs = preprocess_clip(np.array(frames[:cfg.DATA.NUM_FRAMES]), cfg)
    results = predict(model, inputs, class_names)

    print("\nTop-5 predictions:")
    for rank, (name, prob) in enumerate(results, 1):
        print(f"  {rank}. {name:40s}  {prob:.3f}")
```

---

## 9.7 Troubleshooting the Demo

| Problem | Fix |
|---------|-----|
| `KeyError: 'model_state'` | Wrong checkpoint type — try `TRAIN.CHECKPOINT_TYPE caffe2` |
| Black output video | ffmpeg codec issue — try `DEMO.OUTPUT_FILE` with `.avi` extension |
| Very slow on CPU | Normal — use GPU or reduce clip length |
| Wrong predictions | Checkpoint/config mismatch — ensure they correspond to same training |
| `AssertionError: input tensor shape` | NUM_FRAMES mismatch — check config vs checkpoint |

---

## 9.8 Knowledge Check

1. Which model is best for a real-time CPU-only demo and why?
2. What flag do you add when loading a `.pkl` checkpoint?
3. What does `pack_pathway_output` do in the inference script?
4. How would you modify the demo to show only predictions with confidence > 70%?

---

## Next Module

[Module 10 — Hands-On Projects →](10_projects.md)
