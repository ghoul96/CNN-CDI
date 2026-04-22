"""
benchmark_models.py
───────────────────
Measures inference speed (clips/second) for different PySlowFast models
on CPU or GPU.

Usage (after installing PySlowFast):
    python scripts/benchmark_models.py

Prints a table like:
    Model                          Clips/sec   Params      GFLOPs
    ─────────────────────────────────────────────────────────────
    C2D R50                        25.3        35.5 M      ?
    SlowFast R50 (4x16)            12.1        34.0 M      ?
    X3D-M                          47.8         3.8 M      4.73
"""

import time
import torch

# Requires PySlowFast to be installed and PYTHONPATH set
try:
    from slowfast.config.defaults import get_cfg
    from slowfast.models import build_model
except ImportError:
    raise SystemExit(
        "PySlowFast not found.\n"
        "Install it with: cd /path/to/slowfast && python setup.py build develop\n"
        "Then set PYTHONPATH=/path/to/slowfast/slowfast"
    )


CONFIGS = [
    ("C2D R50",              "configs/Kinetics/C2D_8x8_R50.yaml"),
    ("SlowFast R50 (4x16)",  "configs/Kinetics/SLOWFAST_4x16_R50.yaml"),
    ("X3D-M",                "configs/Kinetics/X3D_M.yaml"),
]

N_WARMUP = 10
N_RUNS   = 50
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


def make_dummy_input(cfg) -> list:
    """Create a list of random tensors matching the model's expected input."""
    T    = cfg.DATA.NUM_FRAMES
    crop = cfg.DATA.TEST_CROP_SIZE
    n_pathways = len(cfg.DATA.INPUT_CHANNEL_NUM)

    if n_pathways == 1:
        return [torch.zeros(1, 3, T, crop, crop, device=DEVICE)]

    # SlowFast: two pathways with different temporal lengths
    alpha = cfg.SLOWFAST.ALPHA
    return [
        torch.zeros(1, 3, T,         crop, crop, device=DEVICE),  # Slow
        torch.zeros(1, 3, T * alpha, crop, crop, device=DEVICE),  # Fast
    ]


def benchmark(name: str, cfg_path: str) -> dict:
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.NUM_GPUS = 1 if DEVICE == "cuda" else 0
    cfg.freeze()

    model = build_model(cfg).to(DEVICE)
    model.eval()

    dummy = make_dummy_input(cfg)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(N_RUNS):
            model(dummy)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    clips_per_sec = N_RUNS / elapsed

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return {"name": name, "clips_per_sec": clips_per_sec, "params_m": n_params}


def main():
    print(f"\nDevice: {DEVICE}")
    print(f"Warmup runs: {N_WARMUP}  |  Benchmark runs: {N_RUNS}\n")
    print(f"{'Model':<30}  {'Clips/sec':>10}  {'Params (M)':>10}")
    print("─" * 55)

    for name, cfg_path in CONFIGS:
        try:
            result = benchmark(name, cfg_path)
            print(
                f"{result['name']:<30}  "
                f"{result['clips_per_sec']:>10.1f}  "
                f"{result['params_m']:>10.1f}"
            )
        except Exception as e:
            print(f"{name:<30}  ERROR: {e}")


if __name__ == "__main__":
    main()
