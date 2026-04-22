"""
make_synthetic_dataset.py
─────────────────────────
Generates a tiny synthetic video dataset for testing PySlowFast
without downloading any real data.

Usage:
    python scripts/make_synthetic_dataset.py [--out /tmp/synthetic_kinetics]

Output structure:
    <out>/
    ├── train.csv
    ├── val.csv
    └── videos/
        ├── train/
        │   ├── waving_hand/0000.mp4 ... 0019.mp4
        │   └── ...
        └── val/
            └── ...
"""

import argparse
import os
import pathlib
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python is required: pip install opencv-python")


CLASSES = ["waving_hand", "clapping", "jumping", "cooking", "typing"]
SPLIT_SIZES = {"train": 20, "val": 5}
FPS = 10
DURATION_FRAMES = 30   # 3 seconds at 10 fps
FRAME_SIZE = (224, 224)


def write_video(path: pathlib.Path, label_idx: int):
    """Write a video whose colour is seeded by its label for distinguishability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(label_idx)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, FRAME_SIZE)
    for _ in range(DURATION_FRAMES):
        # Add slight temporal noise so frames differ
        base = rng.integers(40 * label_idx, 40 * label_idx + 40, 3, dtype=np.uint8)
        noise = rng.integers(0, 30, (*FRAME_SIZE, 3), dtype=np.uint8)
        frame = np.clip(base.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()


def main(out_dir: str):
    root = pathlib.Path(out_dir)

    for split, n_per_class in SPLIT_SIZES.items():
        rows = []
        for label, cls_name in enumerate(CLASSES):
            for vid_id in range(n_per_class):
                video_path = root / "videos" / split / cls_name / f"{vid_id:04d}.mp4"
                write_video(video_path, label)
                rows.append(f"{video_path.resolve()} {label}")

        csv_path = root / f"{split}.csv"
        csv_path.write_text("\n".join(rows) + "\n")
        print(f"Wrote {csv_path}  ({len(rows)} clips)")

    # Write a classes.json for demo compatibility
    import json
    classes_path = root / "classnames.json"
    classes_path.write_text(json.dumps({c: i for i, c in enumerate(CLASSES)}, indent=2))
    print(f"Wrote {classes_path}")
    print(f"\nDataset ready at: {root.resolve()}")
    print("Use with:")
    print(f"  DATA.PATH_TO_DATA_DIR {root.resolve()}")
    print(f"  MODEL.NUM_CLASSES {len(CLASSES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic video dataset")
    parser.add_argument("--out", default="/tmp/synthetic_kinetics",
                        help="Output directory (default: /tmp/synthetic_kinetics)")
    args = parser.parse_args()
    main(args.out)
