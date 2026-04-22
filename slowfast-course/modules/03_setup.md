# Module 03 — Environment Setup

## 3.1 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 18.04 / Windows 10 (WSL2) / macOS | Ubuntu 20.04+ |
| Python | 3.8 | 3.9 or 3.10 |
| CUDA | 10.2 | 11.7 or 11.8 |
| GPU VRAM | 8 GB (inference only) | 24 GB (training) |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB (code + small dataset) | 200 GB+ (Kinetics) |

> **CPU-only mode:** You can run inference and short training experiments on CPU
> by setting `NUM_GPUS 0` in any config / command. Training will be very slow.

---

## 3.2 Step 1 — Create a Conda Environment

```bash
# Create and activate a fresh environment
conda create -n slowfast python=3.9 -y
conda activate slowfast
```

Always use a dedicated environment to avoid dependency conflicts with other
projects.

---

## 3.3 Step 2 — Install PyTorch

Visit [pytorch.org](https://pytorch.org/) to get the exact command for your
CUDA version. Below are common examples:

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

Verify the installation:
```python
import torch
print(torch.__version__)          # e.g. 2.1.0
print(torch.cuda.is_available())  # True if GPU found
```

---

## 3.4 Step 3 — Install Core Dependencies

```bash
# fvcore — Facebook's general ML utilities (metrics, logging, config)
pip install 'git+https://github.com/facebookresearch/fvcore'

# iopath — path utilities (local, HDFS, S3)
pip install -U iopath

# simplejson, psutil, OpenCV, TensorBoard
pip install simplejson psutil opencv-python tensorboard

# PyAV — Python bindings for ffmpeg (video decoding)
conda install av -c conda-forge -y

# PyTorchVideo — additional models and augmentations
pip install pytorchvideo

# moviepy — optional, for video rendering in TensorBoard
pip install moviepy

# FairScale — model parallelism utilities
pip install 'git+https://github.com/facebookresearch/fairscale'
```

---

## 3.5 Step 4 — Install Detectron2 (for AVA detection only)

Detectron2 is only needed if you plan to run action **detection** experiments.
Skip this step for classification-only work.

```bash
pip install -U torch torchvision cython
pip install 'git+https://github.com/facebookresearch/fvcore.git' \
            'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

---

## 3.6 Step 5 — Clone and Build PySlowFast

```bash
git clone https://github.com/facebookresearch/slowfast
cd slowfast
python setup.py build develop
```

The `build develop` step compiles any C++ / CUDA extensions in-place so
you can edit the source and immediately see changes without reinstalling.

---

## 3.7 Step 6 — Set PYTHONPATH

PySlowFast requires the `slowfast/` sub-directory to be on your Python path:

```bash
# Linux / macOS
export PYTHONPATH=/absolute/path/to/slowfast/slowfast:$PYTHONPATH

# Windows (PowerShell)
$env:PYTHONPATH = "C:\path\to\slowfast\slowfast;$env:PYTHONPATH"
```

Add the export line to your `~/.bashrc` (or `~/.zshrc`) to make it permanent.

---

## 3.8 Step 7 — Verify the Installation

Run the minimal sanity check:

```bash
python tools/run_net.py --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 2 \
  DATA.PATH_TO_DATA_DIR /tmp/fake_data \
  TRAIN.ENABLE False \
  TEST.ENABLE False
```

If no import errors appear, the installation succeeded. The run will fail when
it tries to load the (non-existent) dataset, but that is expected.

---

## 3.9 Docker Alternative

Facebook provides no official Docker image, but the community maintains one:

```bash
docker pull pytorch/pytorch:2.1.0-cuda11.7-cudnn8-runtime
# Then install dependencies inside the container as above.
```

Using Docker is recommended for reproducible results and easy deployment.

---

## 3.10 Common Installation Errors

| Error | Likely Cause | Fix |
|-------|--------------|-----|
| `No module named 'fvcore'` | fvcore not installed | Run step 3.4 |
| `ImportError: libcuda.so` | CUDA mismatch | Reinstall PyTorch matching your CUDA version |
| `ModuleNotFoundError: slowfast` | PYTHONPATH not set | Run step 3.7 |
| `av.codec.CodecContext` error | PyAV / ffmpeg version | `conda install av=9.2 -c conda-forge` |
| `detectron2` import error | Detectron2 not built | Skip if not using detection |

---

## 3.11 Knowledge Check

1. Why is it recommended to use a Conda environment instead of the system Python?
2. What does `python setup.py build develop` do differently from
   `python setup.py install`?
3. When would you skip the Detectron2 installation?
4. What environment variable must be set so Python can find the `slowfast`
   package?

---

## Next Module

[Module 04 — Datasets & Data Preparation →](04_datasets.md)
