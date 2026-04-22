# Module 01 — Introduction to Video Understanding

## 1.1 What Is Video Understanding?

Video understanding is the field of computer vision that tasks a model with
interpreting sequences of images over time. Unlike a single photo, a video carries:

- **Spatial information** — what objects are in each frame
- **Temporal information** — how those objects move and interact across time

The combination is significantly harder than image classification and much richer in
application scenarios.

---

## 1.2 Core Tasks

| Task | What the model outputs | Example |
|------|------------------------|---------|
| **Action Classification** | A single label for the whole clip | "playing basketball" |
| **Action Detection** | Bounding box + label per person per frame | Person A is "throwing", Person B is "catching" |
| **Temporal Action Localization** | Start/end time + label in a long video | "Typing" starts at 0:04, ends at 0:09 |
| **Video QA** | Free-form or multi-choice answer | "What sport is being played?" → "Tennis" |

PySlowFast focuses on **classification** and **detection** (via the AVA dataset).

---

## 1.3 Why Is Video Hard?

```
Challenge                    Why It Matters
─────────────────────────────────────────────────────────────
High dimensionality          A 1-second 224×224 clip @ 30fps
                             = 30 × 224 × 224 × 3 ≈ 4.5M values

Temporal redundancy          Adjacent frames are nearly identical
                             → naive 3D convnets are expensive

Motion variability           Fast actions (diving) need high fps;
                             slow actions (hand gesture) need context

Background clutter           Static scenes share cues across classes
```

---

## 1.4 A Brief History of Video Models

```
2014  C3D          — First large-scale 3-D convolution network
2016  Two-Stream   — Separate spatial & optical-flow streams fused late
2017  I3D          — Inflate 2-D ImageNet weights → 3-D filters (great baseline)
2017  Non-local    — Self-attention across time, like an early Video Transformer
2019  SlowFast     — Dual pathway: Slow (semantics) + Fast (motion) [ICCV 2019]
2020  X3D          — Progressive expansion for efficiency (CVPR 2020)
2021  MViT         — Multiscale Vision Transformer for video (ICCV 2021)
2022  MViTv2/MAE/  — Improved transformers + masked pre-training
      MaskFeat
2022  Rev-ViT      — Reversible network → huge memory savings
```

All of the above are **implemented and benchmarked in PySlowFast**.

---

## 1.5 Benchmark Datasets

### Kinetics-400 / 600 / 700
- ~240k / 389k / 650k 10-second YouTube clips
- 400 / 600 / 700 human action classes (e.g., "salsa dancing", "answering questions")
- The go-to benchmark for action **classification**
- Top-1 accuracy on K400: humans ≈ 76%, MViTv2-L ≈ 86%

### AVA (Atomic Visual Actions)
- 430 15-minute movie clips
- 80 fine-grained action labels per bounding box per second
- The standard benchmark for spatiotemporal **detection**
- Metric: mean Average Precision (mAP)

### Charades
- 9,848 videos, avg. 30 seconds; activities performed by people at home
- Multi-label, overlapping actions
- Metric: classification mAP

### Something-Something V2 (SSv2)
- 220k short clips of hand-object interactions
- Strongly **temporal** — you must understand motion direction, not just objects

---

## 1.6 How PySlowFast Fits In

PySlowFast is **not** a single model — it is a **research platform** that bundles:

1. Multiple state-of-the-art backbone families
2. A unified training/evaluation pipeline (`tools/run_net.py`)
3. A YAML-driven config system for reproducible experiments
4. Pre-trained model checkpoints (Model Zoo)
5. Visualization and demo tools

Think of it as "the ImageNet training scripts of video understanding".

---

## 1.7 Knowledge Check

1. Name three differences between image classification and video classification.
2. What dataset would you use to train an action detection model?
3. Why is the SlowFast architecture inspired by biology (hint: look up the
   primate visual cortex P-cells and M-cells)?
4. What does "Top-1 accuracy" mean on Kinetics-400?

---

## Next Module

[Module 02 — SlowFast Architecture Deep-Dive →](02_architecture.md)
