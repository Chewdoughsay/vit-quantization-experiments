"""
Phase 4 — Native FP8 evaluation via torch.float8_e4m3fn.

Compares:
  - FP32 baseline
  - INT8 per-tensor (software, our implementation from Phase 2)
  - INT8 per-channel (software, our implementation from Phase 3)
  - FP8 E4M3FN per-tensor  (torch native dtype)
  - FP8 E4M3FN per-channel (torch native dtype)

Hardware requirements:
  - H100  (Hopper)  — hardware FP8 GEMM, full speedup
  - A100  (Ampere)  — FP8 dtype supported, some ops accelerated
  - Other — FP8 simulated in software (no speedup, for accuracy only)

Torch version: 2.1+ required for float8_e4m3fn

Usage:
    python scripts/phase4/evaluate_fp8_native.py
    python scripts/phase4/evaluate_fp8_native.py --model vit_small_patch16_224
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import timm
import timm.data
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.quantized_linear import (
    quantize_model_selective,
    quantize_model_per_channel,
    SKIP_PATTERNS,
)

# ═══════════════════════════════════════════════════════════════════════════
# FP8 quantization (native torch dtype)
# ═══════════════════════════════════════════════════════════════════════════

FP8_DTYPE   = torch.float8_e4m3fn
FP8_MAX     = 448.0   # max representable value for e4m3fn


def fp8_quantize_per_tensor(w: torch.Tensor):
    """Quantize weight to FP8 E4M3FN (per-tensor scale)."""
    scale = w.abs().max() / FP8_MAX
    scale = scale.clamp(min=1e-12)
    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale


def fp8_dequantize(q: torch.Tensor, scale) -> torch.Tensor:
    """Dequantize FP8 back to float32."""
    return q.to(torch.float32) * scale


def fp8_quantize_per_channel(w: torch.Tensor):
    """Quantize weight to FP8 E4M3FN (per output-channel scale)."""
    scales = w.abs().max(dim=1, keepdim=True).values / FP8_MAX
    scales = scales.clamp(min=1e-12)
    q = (w / scales).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scales


class FP8Linear(nn.Module):
    """Drop-in nn.Linear replacement storing weights in FP8 E4M3FN."""

    def __init__(self, q_weight, scale, bias=None, per_channel: bool = False):
        super().__init__()
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.bias       = bias
        self.per_channel = per_channel

    @classmethod
    def from_linear(cls, module: nn.Linear, per_channel: bool = False):
        w = module.weight.data
        if per_channel:
            q, scale = fp8_quantize_per_channel(w)
        else:
            q, scale = fp8_quantize_per_tensor(w)
        return cls(q, scale, module.bias, per_channel=per_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fp8_dequantize(self.q_weight, self.scale)
        return nn.functional.linear(x, w, self.bias)


def _set_nested(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def quantize_model_fp8(
    model: nn.Module,
    per_channel: bool = False,
    skip_patterns=None,
    verbose: bool = False,
) -> tuple[nn.Module, list[dict]]:
    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS

    layer_stats = []
    targets = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]
    for name, module in targets:
        if any(p in name for p in skip_patterns):
            continue
        w = module.weight.data
        if per_channel:
            q, scales = fp8_quantize_per_channel(w)
            w_rec = fp8_dequantize(q, scales)
        else:
            q, scale = fp8_quantize_per_tensor(w)
            w_rec = fp8_dequantize(q, scale)

        mse = float(((w - w_rec) ** 2).mean())
        layer_stats.append({"layer": name, "mse": mse})
        _set_nested(model, name, FP8Linear.from_linear(module, per_channel=per_channel))

    if verbose:
        for s in layer_stats:
            print(f"  {s['layer']:<50} MSE={s['mse']:.2e}")
    return model, layer_stats


# ═══════════════════════════════════════════════════════════════════════════
# Device helpers (same as evaluate_model.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ═══════════════════════════════════════════════════════════════════════════
# DataLoader
# ═══════════════════════════════════════════════════════════════════════════

class HFImageNet(Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            import numpy as np
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), item["label"]


def load_imagenet_val(transform, batch_size: int = 64,
                      data_dir: str = "data/imagenet-1k") -> DataLoader:
    from datasets import load_dataset
    local_path = Path(data_dir)
    if local_path.exists() and any(local_path.glob("*.parquet")):
        print(f"  Loading from local parquet: {local_path}")
        hf_ds = load_dataset("parquet",
                             data_files=str(local_path / "*.parquet"),
                             split="train")
    else:
        print("  Downloading from HuggingFace...")
        hf_ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                             trust_remote_code=True)
    print(f"  {len(hf_ds):,} images\n")
    device = get_device()
    pin = device.type == "cuda"
    workers = 4 if device.type == "cuda" else 2
    return DataLoader(HFImageNet(hf_ds, transform), batch_size=batch_size,
                      shuffle=False, num_workers=workers,
                      pin_memory=pin, persistent_workers=(workers > 0))


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, label="", warmup=3) -> dict:
    model.eval()
    is_half = next(model.parameters()).dtype == torch.float16
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    latencies = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc=label, leave=True)):
            if is_half:
                images = images.half()
            images, labels = images.to(device), labels.to(device)
            sync_device(device)
            t0 = time.perf_counter()
            outputs = model(images)
            sync_device(device)
            t1 = time.perf_counter()
            loss = criterion(outputs, labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
            if i >= warmup:
                latencies.append((t1 - t0) * 1000)

    return {
        "accuracy":                 round(correct / total * 100, 3),
        "avg_loss":                 round(total_loss / len(loader), 6),
        "avg_latency_ms_per_batch": round(float(np.mean(latencies)), 3),
        "std_latency_ms":           round(float(np.std(latencies)), 3),
        "total_samples":            total,
    }


def model_size_mb(model) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total / 1024 ** 2, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "FP32":    "#0072B2",
    "FP16":    "#009E73",
    "INT8-pt": "#E69F00",
    "INT8-pc": "#D55E00",
    "FP8-pt":  "#CC79A7",
    "FP8-pc":  "#56B4E9",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "#F8F8F8",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
})


def generate_plots(results: dict, plots_dir: Path, fp32_acc: float) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in results if m not in ("model", "timestamp",
               "pytorch", "device", "batch_size", "fp8_available")]

    accs     = [results[m]["accuracy"] for m in methods]
    mems     = [results[m]["memory_mb"] for m in methods]
    lats     = [results[m]["avg_latency_ms_per_batch"] for m in methods]
    deltas   = [acc - fp32_acc for acc in accs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'FP8 Native vs INT8 — {results["model"]}',
                 fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in zip(
        axes,
        [accs, mems, lats],
        ["Top-1 Accuracy (%)", "Memory (MB)", "Latency (ms/batch)"],
        ["Accuracy (%)", "MB", "ms/batch"],
    ):
        ax.bar(methods, vals, color=[COLORS.get(m, "#999") for m in methods], width=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(methods, rotation=25, ha="right")

    plt.tight_layout()
    fig.savefig(plots_dir / "00_fp8_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Degradation bar
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(methods, deltas, color=[COLORS.get(m, "#999") for m in methods], width=0.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Accuracy Degradation vs FP32")
    ax.set_ylabel("Δ Accuracy (pp)")
    ax.set_xticklabels(methods, rotation=25, ha="right")
    for i, (m, d) in enumerate(zip(methods, deltas)):
        ax.text(i, d + (0.002 if d >= 0 else -0.005),
                f"{d:+.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(plots_dir / "01_degradation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_tiny_patch16_224")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data/imagenet-1k")
    args = parser.parse_args()

    device = get_device()
    fp8_available = hasattr(torch, "float8_e4m3fn")

    print("=" * 70)
    print(f"Phase 4 — FP8 Native Evaluation — {args.model}")
    print(f"PyTorch {torch.__version__}  |  device: {device}")
    print(f"FP8 (float8_e4m3fn) available: {fp8_available}")
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU: {torch.cuda.get_device_name()}  |  "
              f"Compute capability: {cap[0]}.{cap[1]}")
        if cap[0] >= 9:
            print("  → Hopper GPU: hardware FP8 GEMM available")
        elif cap[0] >= 8:
            print("  → Ampere GPU: FP8 dtype supported, partial acceleration")
    print("=" * 70)

    results_dir = Path(f"results/Phase4/{args.model}/fp8_native")
    metrics_dir = results_dir / "metrics"
    plots_dir   = results_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_model = timm.create_model(args.model, pretrained=True, num_classes=1000)
    data_cfg   = timm.data.resolve_data_config({}, model=base_model)
    transform  = timm.data.create_transform(**data_cfg)

    print("\nLoading ImageNet-1k validation...")
    loader = load_imagenet_val(transform, batch_size=args.batch_size,
                               data_dir=args.data_dir)

    results = {
        "model": args.model, "device": str(device),
        "pytorch": torch.__version__,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fp8_available": fp8_available,
        "batch_size": args.batch_size,
    }

    configs = [
        ("FP32",    lambda m: m,                                               "FP32 baseline"),
        ("INT8-pt", lambda m: quantize_model_selective(m, verbose=False)[0],   "INT8 per-tensor"),
        ("INT8-pc", lambda m: quantize_model_per_channel(m, verbose=False)[0], "INT8 per-channel"),
    ]
    if fp8_available:
        configs += [
            ("FP8-pt", lambda m: quantize_model_fp8(m, per_channel=False, verbose=False)[0], "FP8 per-tensor"),
            ("FP8-pc", lambda m: quantize_model_fp8(m, per_channel=True, verbose=False)[0],  "FP8 per-channel"),
        ]
    else:
        print("\nWARNING: torch.float8_e4m3fn not available — FP8 configs skipped.")

    for label, quantize_fn, desc in configs:
        print(f"\n{'='*70}\n{label}: {desc}\n{'='*70}")
        model = copy.deepcopy(base_model).to(device)
        model = quantize_fn(model)
        mem = model_size_mb(model)
        res = evaluate(model, loader, device, label=label)
        results[label] = {**res, "memory_mb": mem}
        print(f"  Accuracy: {res['accuracy']}%  |  "
              f"Latency: {res['avg_latency_ms_per_batch']} ms  |  Memory: {mem} MB")
        del model

    fp32_acc = results["FP32"]["accuracy"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Accuracy':>10} {'Δ FP32':>10} {'Mem (MB)':>10} {'Lat (ms)':>12}")
    print("-" * 60)
    for cfg_label, _, _ in configs:
        r = results[cfg_label]
        delta = r["accuracy"] - fp32_acc
        print(f"{cfg_label:<12} {r['accuracy']:>9.3f}%  "
              f"{delta:>+9.4f}  {r['memory_mb']:>9.2f}  "
              f"{r['avg_latency_ms_per_batch']:>11.1f}")

    out_path = metrics_dir / "fp8_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    generate_plots(results, plots_dir, fp32_acc)
    print("=" * 70)


if __name__ == "__main__":
    main()
