"""
make_kinetics_csv.py
────────────────────
Generates train.csv / val.csv / test.csv for a Kinetics-style dataset
that is organised as:

    root/
    ├── train/
    │   ├── class_name_0/
    │   │   ├── video1.mp4
    │   │   └── ...
    │   └── class_name_1/ ...
    ├── val/  (same structure)
    └── test/ (same structure)

Usage:
    python scripts/make_kinetics_csv.py --root /data/kinetics400 \
                                        --out  /data/kinetics400

The script writes <out>/train.csv, val.csv, test.csv.  Labels are
assigned alphabetically so they are stable across runs.
"""

import argparse
import pathlib


def build_csv(root: pathlib.Path, split: str, class_to_idx: dict) -> list[str]:
    split_dir = root / split
    if not split_dir.is_dir():
        print(f"  Warning: {split_dir} does not exist, skipping.")
        return []

    rows = []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = class_to_idx.get(cls_dir.name)
        if label is None:
            print(f"  Warning: class '{cls_dir.name}' not in class list, skipping.")
            continue
        for video in sorted(cls_dir.glob("*.mp4")):
            rows.append(f"{video.resolve()} {label}")
        for video in sorted(cls_dir.glob("*.webm")):
            rows.append(f"{video.resolve()} {label}")
    return rows


def main(root: str, out: str):
    root_path = pathlib.Path(root)
    out_path  = pathlib.Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build class list from the training split
    train_dir = root_path / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"ERROR: {train_dir} not found. "
                         "Make sure the dataset is organised as root/train/class/video.mp4")

    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(classes)} classes.")

    for split in ("train", "val", "test"):
        rows = build_csv(root_path, split, class_to_idx)
        if rows:
            csv_path = out_path / f"{split}.csv"
            csv_path.write_text("\n".join(rows) + "\n")
            print(f"Wrote {csv_path}  ({len(rows)} clips)")

    # Write class name mapping for reference
    import json
    mapping_path = out_path / "class_mapping.json"
    mapping_path.write_text(json.dumps(class_to_idx, indent=2))
    print(f"Wrote {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kinetics CSV files")
    parser.add_argument("--root", required=True, help="Root directory of the dataset")
    parser.add_argument("--out",  required=True, help="Directory to write CSV files")
    args = parser.parse_args()
    main(args.root, args.out)
