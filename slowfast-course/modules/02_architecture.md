# Module 02 — SlowFast Architecture Deep-Dive

## 2.1 The Core Idea: Two Pathways

The SlowFast paper (Feichtenhofer et al., ICCV 2019, arXiv:1812.03982) draws
inspiration from the primate visual system:

- **Parvocellular (P) cells** — fire slowly, high spatial resolution, colour,
  detail → **Slow Pathway**
- **Magnocellular (M) cells** — fire fast, low spatial resolution, high
  temporal resolution, motion → **Fast Pathway**

```
Input Video Frames
        │
        ├──── Sample every α frames ──► Slow Pathway (low fps, high channels)
        │                                    │
        └──── Sample every 1 frame  ──► Fast Pathway (high fps, low channels)
                                             │
                                    Lateral Connections (fast → slow)
                                             │
                                     Fused Representation
                                             │
                                     Classification Head
```

### Key Hyper-parameters

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| `T` | Number of frames sampled for Slow pathway | 4 or 8 |
| `α` (alpha) | Frame-rate ratio (Fast runs α× faster) | 8 |
| `β` (beta) | Channel ratio (Fast has β× fewer channels) | 1/8 |
| `τ` (tau) | Temporal stride between sampled frames | 16 |

So if `T=4, α=8, τ=16`:
- **Slow pathway** sees 4 frames spaced 16 frames apart → 64 raw frames, low fps
- **Fast pathway** sees `T×α = 32` frames spaced 2 frames apart → high fps

---

## 2.2 Slow Pathway

- Standard ResNet/Transformer backbone (ResNet-50, ResNet-101, etc.)
- **Low frame rate** → captures appearance and spatial semantics well
- **High channel capacity** → `β` fraction of channels go to Fast
- Uses temporally **strided** 3-D convolutions

```python
# Conceptually (pseudo-code from slowfast/models/video_model_builder.py)
slow_frames = all_frames[::alpha]          # sparse sampling
slow_features = ResNet_3D(slow_frames)     # full channel width
```

---

## 2.3 Fast Pathway

- Lightweight version of the same backbone with only `β` of the channels
- **High frame rate** → captures fine-grained temporal/motion cues
- **Fewer channels** → keeps computation proportional to Slow pathway

```python
fast_frames = all_frames                   # dense sampling
fast_features = ResNet_3D_light(fast_frames)  # β fraction of channels
```

Because β = 1/8, the Fast pathway uses only ~20% of the total FLOPs despite
processing 8× more frames. This is the elegant efficiency trade-off.

---

## 2.4 Lateral Connections

The Fast pathway sends information to the Slow pathway at each stage via
**lateral connections** (unidirectional: Fast → Slow):

```
Fast feature map  ──[Time-strided conv or concat]──► Slow feature map
```

Three fusion types are supported:
- **Time-to-channel (TtoC)** — reshape temporal dim into channel dim
- **Concatenation** — stack feature maps along channel
- **Sum** — element-wise addition

The lateral connections allow the Slow pathway to be informed by motion
information even though it processes fewer frames.

---

## 2.5 Other Supported Architectures

### C2D
A 2-D ResNet that pools temporal information at the very end. Simplest baseline.
Good for ablation studies.

### I3D (Inflated 3D ConvNet)
Takes a 2-D ImageNet-pretrained ResNet and "inflates" all 2-D convolutions to
3-D equivalents by replicating weights along the temporal axis. The inflation
trick allows pre-training benefit from ImageNet.

### Non-local Network
Adds self-attention blocks (non-local means) at specific layers of I3D. Captures
long-range space-time dependencies that local convolutions miss.

### X3D (Progressive Network Expansion)
Starts from a tiny 2-D network and progressively expands one dimension at a time:
- X: spatial resolution
- 3: temporal duration (clip length)
- D: depth (number of layers), width (channels), and fps

Result: a family of extremely efficient models (X3D-XS to X3D-L).

| Model | GFLOPs | Params | K400 Top-1 |
|-------|--------|--------|-----------|
| X3D-XS | 0.60 | 3.8M | 68.7% |
| X3D-S | 1.96 | 3.8M | 73.1% |
| X3D-M | 4.73 | 3.8M | 75.1% |
| X3D-L | 18.37 | 6.2M | 76.9% |

### MViT (Multiscale Vision Transformer)
Applies the Vision Transformer (ViT) idea to video with **multiscale pooling**:
- Early stages: high resolution, few channels
- Later stages: low resolution, many channels
- Hierarchical structure mimics CNNs inside a transformer

MViTv1-B achieves 78.4% on K400; MViTv2-B achieves 82.9%.

### Rev-ViT / Rev-MViT
Reversible computation means **activations do not need to be stored** during
forward pass for back-prop — the backward pass recomputes them. This slashes
GPU memory usage dramatically, enabling larger batch sizes or longer clips.

---

## 2.6 Architecture Comparison Table

| Model | Type | Temporal Modeling | Memory | Accuracy (K400) |
|-------|------|-------------------|--------|-----------------|
| C2D | 2-D CNN + pool | Late temporal pooling | Low | 67.2% |
| I3D | 3-D CNN | Inflated 3-D conv | High | 73.5% |
| I3D + NLN | 3-D CNN + Attn | Non-local self-attention | Very High | 74.0% |
| Slow | 3-D CNN | Sparse sampling | Medium | 74.8% |
| SlowFast | Dual-path 3-D | Slow + Fast + Lateral | Medium | 77.0% |
| X3D-M | Compact 3-D | Progressive expansion | Low | 75.1% |
| MViTv1-B | Transformer | Multiscale attention | Medium | 78.4% |
| MViTv2-B | Transformer | Improved MS attention | Medium | 82.9% |

---

## 2.7 How a Forward Pass Works (Step-by-Step)

```
1. VIDEO CLIP LOADING
   Raw video  →  decode with PyAV/ffmpeg  →  tensor of shape [C, T_total, H, W]

2. FRAME SAMPLING
   Slow:  pick T frames uniformly from clip
   Fast:  pick α×T frames uniformly from clip

3. DATA AUGMENTATION
   Random short-side scaling, random crop, horizontal flip, colour jitter

4. SLOW PATHWAY
   [C, T, H, W]  →  3-D ResNet stem  →  res2  →  res3  →  res4  →  res5
                                                     ↑ lateral from Fast

5. FAST PATHWAY
   [C, α×T, H, W]  →  lightweight 3-D ResNet (β channels) at each stage
                                                     ↓ sends lateral to Slow

6. HEAD
   Global average pool (both pathways)  →  concatenate  →  Dropout  →  FC  →  softmax

7. OUTPUT
   Class probabilities of shape [batch_size, num_classes]
```

---

## 2.8 Knowledge Check

1. What is the purpose of the Fast pathway if it has fewer channels?
2. Why are lateral connections only Fast → Slow and not bidirectional?
3. If T=8 and α=8, how many total frames does the model need to sample from
   the raw video?
4. What is the key memory-saving trick in Rev-ViT?
5. You need to deploy a model on a mobile device. Which architecture would you
   choose and why?

---

## Next Module

[Module 03 — Environment Setup →](03_setup.md)
