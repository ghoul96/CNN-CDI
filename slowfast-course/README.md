# PySlowFast: Beginner's Course in Video Understanding

**Based on:** [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)  
**Authors of PySlowFast:** Haoqi Fan, Yanghao Li, Bo Xiong, Wan-Yen Lo, Christoph Feichtenhofer (FAIR)  
**License:** Apache 2.0

---

## What You Will Learn

This course takes you from zero to running state-of-the-art video understanding
models using PySlowFast. By the end you will be able to:

- Explain the SlowFast dual-pathway architecture and why it works
- Install and configure the PySlowFast environment
- Prepare datasets (Kinetics, AVA, Charades, SSv2)
- Train models from scratch or fine-tune pretrained checkpoints
- Evaluate accuracy on a held-out test set
- Visualize training metrics, feature maps, and Grad-CAM attention
- Run real-time action recognition and detection demos on your own videos

---

## Course Modules

| # | Module | Topics |
|---|--------|--------|
| 01 | [Introduction to Video Understanding](modules/01_introduction.md) | What is video understanding, tasks, benchmarks |
| 02 | [SlowFast Architecture Deep-Dive](modules/02_architecture.md) | Slow/Fast pathways, lateral connections, supported backbones |
| 03 | [Environment Setup](modules/03_setup.md) | Conda, PyTorch, all dependencies, build |
| 04 | [Datasets & Data Preparation](modules/04_datasets.md) | Kinetics, AVA, Charades, SSv2, CSV format |
| 05 | [Config System](modules/05_config.md) | YAML configs, key parameters explained |
| 06 | [Training Your First Model](modules/06_training.md) | Single-GPU quickstart, multigrid, checkpointing |
| 07 | [Evaluation & Testing](modules/07_evaluation.md) | Metrics, test-time augmentation, test scripts |
| 08 | [Visualization Tools](modules/08_visualization.md) | TensorBoard, Grad-CAM, feature maps, demo |
| 09 | [Running the Demo](modules/09_demo.md) | Inference on wild video, webcam, AVA format |
| 10 | [Hands-On Projects](modules/10_projects.md) | Three guided projects to solidify knowledge |

---

## Prerequisites

- Basic Python (loops, functions, classes)
- Basic familiarity with PyTorch tensors and `nn.Module`
- A machine with a CUDA-capable GPU (≥8 GB VRAM recommended for training;
  CPU-only is sufficient for inference experiments)

---

## Quick-Start (< 5 minutes to first prediction)

```bash
# 1. Clone repo
git clone https://github.com/facebookresearch/slowfast
cd slowfast

# 2. Install (see Module 03 for full details)
pip install -e .

# 3. Download a pretrained checkpoint (X3D-M, ~5 MB)
#    See Model Zoo in Module 07 for all links

# 4. Run inference demo on a sample video
python tools/run_net.py \
  --cfg configs/Kinetics/X3D_M.yaml \
  DEMO.ENABLE True \
  DEMO.INPUT_VIDEO path/to/your_video.mp4 \
  DEMO.LABEL_FILE_PATH kinetics_classnames.json \
  TEST.CHECKPOINT_FILE_PATH path/to/X3D_M.pkl \
  TRAIN.ENABLE False
```

---

## Repository Structure (Key Folders)

```
SlowFast/
├── configs/          # YAML config files for every model × dataset
│   ├── Kinetics/
│   ├── AVA/
│   └── SSv2/
├── slowfast/
│   ├── config/       # defaults.py — every tuneable knob explained
│   ├── datasets/     # Kinetics, AVA, Charades, SSv2 loaders + DATASET.md
│   ├── models/       # All backbone architectures
│   │   ├── video_model_builder.py
│   │   ├── slowfast.py
│   │   ├── x3d.py
│   │   └── mvit.py
│   ├── utils/        # Metrics, logging, checkpointing
│   └── visualization/
├── tools/
│   └── run_net.py    # Single entry-point for train/val/test/demo
├── projects/         # Sub-projects: X3D, MViT, MaskFeat, MAE, ...
├── INSTALL.md
├── GETTING_STARTED.md
├── MODEL_ZOO.md
└── VISUALIZATION_TOOLS.md
```
