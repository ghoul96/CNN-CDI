# Module 04 — Datasets & Data Preparation

## 4.1 How PySlowFast Loads Data

PySlowFast uses a **CSV-driven** data loading approach. For most datasets you
provide three CSV files:

```
train.csv
val.csv
test.csv
```

Each row contains the **path to a video file** (or frame folder) and the
corresponding **integer label**:

```
/data/kinetics/videos/playing_guitar/abc123.mp4 123
/data/kinetics/videos/doing_yoga/xyz456.mp4 256
```

> **Label integers** must correspond to a contiguous range starting at 0.

---

## 4.2 Kinetics (K400 / K600 / K700)

### 4.2.1 Download

Google DeepMind no longer hosts the raw videos directly. Use the ActivityNet
crawler scripts:

```bash
git clone https://github.com/activitynet/ActivityNet
cd ActivityNet/Crawler/Kinetics
pip install -r requirements.txt
python download.py --classes all --output /data/kinetics400
```

> Downloading all of K400 takes many hours and ~140 GB. For learning purposes,
> start with a **mini subset** (see Section 4.6).

### 4.2.2 Resize

All training/inference uses the short edge resized to 256 px:

```bash
# Bash (Linux)
for f in /data/kinetics400/videos_raw/**/*.mp4; do
  out="${f/videos_raw/videos_256}"
  mkdir -p "$(dirname "$out")"
  ffmpeg -i "$f" -vf "scale=-1:256" -q:v 1 "$out" -y
done
```

### 4.2.3 Generate CSV Files

```python
# scripts/make_kinetics_csv.py  (provided in course repository)
import os, pathlib

root = pathlib.Path("/data/kinetics400/videos_256")
splits = {"train": [], "val": [], "test": []}

# Kinetics is organised as  split/classname/video.mp4
classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
class_to_idx = {c: i for i, c in enumerate(classes)}

for split in splits:
    for cls_dir in (root / split).iterdir():
        label = class_to_idx[cls_dir.name]
        for video in cls_dir.glob("*.mp4"):
            splits[split].append(f"{video} {label}")

for split, rows in splits.items():
    with open(f"/data/kinetics400/{split}.csv", "w") as f:
        f.write("\n".join(rows))
```

---

## 4.3 AVA (Action Detection)

AVA is more complex because it includes **person bounding boxes** and
**frame-level labels** rather than clip-level labels.

### Directory Structure Required by PySlowFast

```
ava/
├── frames/
│   ├── video_name_0/
│   │   ├── video_name_0_000001.jpg
│   │   ├── video_name_0_000002.jpg
│   │   └── ...
│   └── video_name_1/ ...
├── frame_lists/
│   ├── train.csv
│   └── val.csv
└── annotations/
    ├── ava_train_v2.1.csv
    ├── ava_val_v2.1.csv
    ├── ava_action_list_v2.1_for_activitynet_2018.pbtxt
    ├── ava_train_predicted_boxes.csv
    └── ava_val_predicted_boxes.csv
```

### Download Steps

```bash
# 1. Download videos (train+val, ~15 min segment per video)
wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt
while IFS= read -r line; do
  wget "https://s3.amazonaws.com/ava-dataset/trainval/$line" -P /data/ava/videos/
done < ava_file_names_trainval_v2.1.txt

# 2. Trim each video to its 15–30 min window (AVA is annotated in this window)
for video in /data/ava/videos/*.mp4; do
  out="/data/ava/videos_trim/${video##*/}"
  ffmpeg -ss 900 -t 901 -i "$video" "$out" -y
done

# 3. Extract frames at 30 fps
for video in /data/ava/videos_trim/*.mp4; do
  name=$(basename "$video" .mp4)
  mkdir -p "/data/ava/frames/$name"
  ffmpeg -i "$video" -r 30 -q:v 1 "/data/ava/frames/$name/${name}_%06d.jpg"
done

# 4. Download annotations
DATA_DIR="/data/ava/annotations"
mkdir -p $DATA_DIR
wget https://research.google.com/ava/download/ava_train_v2.1.csv          -P $DATA_DIR
wget https://research.google.com/ava/download/ava_val_v2.1.csv            -P $DATA_DIR
wget https://research.google.com/ava/download/ava_action_list_v2.1_for_activitynet_2018.pbtxt -P $DATA_DIR

# 5. Download frame lists and predicted boxes
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/train.csv \
     -P /data/ava/frame_lists/
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/frame_lists/val.csv \
     -P /data/ava/frame_lists/
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_train_predicted_boxes.csv \
     -P $DATA_DIR
wget https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_val_predicted_boxes.csv \
     -P $DATA_DIR
```

---

## 4.4 Charades

```bash
# 1. Download RGB frames (~56 GB)
wget http://ai2-website.s3.amazonaws.com/data/Charades_v1_rgb.tar
tar xf Charades_v1_rgb.tar -C /data/charades/

# 2. Download frame lists
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/train.csv \
     -P /data/charades/frame_lists/
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/charades/frame_lists/val.csv \
     -P /data/charades/frame_lists/
```

In your config:
```yaml
DATA:
  PATH_TO_DATA_DIR: /data/charades/frame_lists
  PATH_PREFIX: /data/charades/Charades_v1_rgb
```

---

## 4.5 Something-Something V2 (SSv2)

```bash
# Download from the official website — requires free registration:
# https://20bn.com/datasets/something-something

# After downloading, extract frames at 30 fps:
ffmpeg -i "$video" -r 30 -q:v 1 "frames/{video_name}_%06d.jpg"

# Download frame lists:
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/train.csv
wget https://dl.fbaipublicfiles.com/pyslowfast/dataset/ssv2/frame_lists/val.csv
```

---

## 4.6 Mini Dataset for Learning (No Download Required)

For practice without downloading hundreds of GB, use **PyTorchVideo's demo
dataset** or create a tiny dummy dataset:

### Option A — PyTorchVideo Kinetics Demo

```python
from pytorchvideo.data import Kinetics
# Downloads a 5-video demo split automatically
dataset = Kinetics(
    data_path="demo",
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", 2),
    decode_audio=False,
)
```

### Option B — Synthetic Video Dataset (pure Python)

```python
# scripts/make_synthetic_dataset.py
import os, cv2, numpy as np, random

CLASSES = ["waving_hand", "clapping", "jumping", "cooking", "typing"]
SPLIT_SIZES = {"train": 20, "val": 5}
OUT_DIR = "/tmp/synthetic_kinetics"

for split, n in SPLIT_SIZES.items():
    rows = []
    for label, cls in enumerate(CLASSES):
        cls_dir = f"{OUT_DIR}/videos/{split}/{cls}"
        os.makedirs(cls_dir, exist_ok=True)
        for vid_id in range(n):
            path = f"{cls_dir}/{vid_id:04d}.mp4"
            # Write 30 random RGB frames at 10 fps
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224)
            )
            for _ in range(30):
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            rows.append(f"{path} {label}")
    with open(f"{OUT_DIR}/{split}.csv", "w") as f:
        f.write("\n".join(rows))

print("Synthetic dataset ready at", OUT_DIR)
```

Run and use:
```bash
python scripts/make_synthetic_dataset.py

python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR /tmp/synthetic_kinetics \
  NUM_GPUS 0 \
  TRAIN.BATCH_SIZE 2 \
  MODEL.NUM_CLASSES 5
```

---

## 4.7 Data Augmentation Pipeline

PySlowFast applies these augmentations at **training** time:
1. Random short-side resize (min_scale to max_scale)
2. Random crop to `crop_size × crop_size`
3. Random horizontal flip
4. Colour jitter (brightness, contrast, saturation)
5. MixUp (optional, enabled per config)

At **test** time it uses:
1. Short-side resize to `test_scale`
2. Three uniformly-spaced temporal crops
3. Ten spatial crops (uniform grid + centre)

---

## 4.8 Knowledge Check

1. What two columns are expected in a Kinetics-format CSV file?
2. Why does AVA require bounding box annotations that Kinetics does not?
3. Write a one-liner bash command to resize a single `.mp4` to short-edge 256px.
4. What is the advantage of the synthetic dataset approach for development?

---

## Next Module

[Module 05 — Config System →](05_config.md)
