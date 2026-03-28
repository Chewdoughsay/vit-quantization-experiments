"""
Phase 4 — Cross-model scaling evaluation.

Runs Phases 1-3 equivalent for any ViT model size (Tiny / Small / Base)
on any device (CUDA A100, MPS, CPU).

Models:
    vit_tiny_patch16_224   — 5.7M  params  (already done on M4)
    vit_small_patch16_224  — 22M   params
    vit_base_patch16_224   — 86M   params

Outputs:
    results/Phase4/<model_name>/metrics/results.json
    results/Phase4/<model_name>/plots/

Usage:
    python scripts/phase4/evaluate_model.py --model vit_small_patch16_224
    python scripts/phase4/evaluate_model.py --model vit_base_patch16_224
    python scripts/phase4/evaluate_model.py --model vit_tiny_patch16_224 --batch-size 128
"""

import argparse
import copy
import json
import sys
import tempfile
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.quantized_linear import (
    quantize_model_selective,
    quantize_model_per_channel,
    SKIP_PATTERNS,
)

# ═══════════════════════════════════════════════════════════════════════════
# Device helpers
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
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), item["label"]


def load_imagenet_val(
    transform,
    batch_size: int = 64,
    data_dir: str = "data/imagenet-1k",
) -> DataLoader:
    from datasets import load_dataset

    local_path = Path(data_dir)
    if local_path.exists() and any(local_path.glob("*.parquet")):
        print(f"  Loading from local parquet: {local_path}")
        hf_ds = load_dataset(
            "parquet",
            data_files=str(local_path / "*.parquet"),
            split="train",
        )
    else:
        print("  Downloading ImageNet-1k validation from HuggingFace...")
        hf_ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            trust_remote_code=True,
        )
    print(f"  {len(hf_ds):,} images\n")

    device = get_device()
    pin = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 2

    return DataLoader(
        HFImageNet(hf_ds, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label: str = "",
    warmup: int = 3,
) -> dict:
    model.eval()
    is_half = next(model.parameters()).dtype == torch.float16

    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    latencies: list[float] = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc=label, leave=True)):
            if is_half:
                images = images.half()
            images = images.to(device)
            labels = labels.to(device)

            sync_device(device)
            t0 = time.perf_counter()

            outputs = model(images)

            sync_device(device)
            t1 = time.perf_counter()

            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

            if i >= warmup:
                latencies.append((t1 - t0) * 1000)

    accuracy = correct / total
    return {
        "accuracy":                accuracy,
        "accuracy_percent":        round(accuracy * 100, 3),
        "avg_loss":                round(total_loss / len(loader), 6),
        "avg_latency_ms_per_batch": round(float(np.mean(latencies)), 3),
        "std_latency_ms":          round(float(np.std(latencies)), 3),
        "total_samples":           total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Memory helpers
# ═══════════════════════════════════════════════════════════════════════════

def model_size_mb(model: nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total / (1024 ** 2), 3)


def disk_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return round(Path(f.name).stat().st_size / (1024 ** 2), 3)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "FP32":    "#0072B2",
    "FP16":    "#009E73",
    "INT8-pt": "#E69F00",
    "INT8-pc": "#D55E00",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "legend.fontsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "#F8F8F8",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
})


def generate_plots(results: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    methods = ["FP32", "FP16", "INT8-pt", "INT8-pc"]
    accs    = [results[m]["accuracy_percent"] for m in methods]
    mems    = [results[m]["memory_mb"] for m in methods]
    lats    = [results[m]["avg_latency_ms_per_batch"] for m in methods]
    fp32_acc = results["FP32"]["accuracy_percent"]

    # --- Summary ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Phase 4 Summary — {results["model"]}', fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in zip(
        axes,
        [accs, mems, lats],
        ["Accuracy", "Memory (MB)", "Latency (ms/batch)"],
        ["Top-1 Accuracy (%)", "Memory (MB)", "Latency (ms/batch)"],
    ):
        bars = ax.bar(methods, vals, color=[COLORS[m] for m in methods], width=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(plots_dir / "00_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Accuracy degradation ---
    fig, ax = plt.subplots(figsize=(8, 5))
    deltas = [acc - fp32_acc for acc in accs]
    bars = ax.bar(methods, deltas, color=[COLORS[m] for m in methods], width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Accuracy Degradation vs FP32")
    ax.set_ylabel("Δ Accuracy (pp)")
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.001 if val >= 0 else -0.003),
                f"{val:+.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(plots_dir / "01_accuracy_degradation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

MODELS = {
    "vit_tiny_patch16_224":  {"params": "5.7M",  "embed_dim": 192},
    "vit_small_patch16_224": {"params": "22M",   "embed_dim": 384},
    "vit_base_patch16_224":  {"params": "86M",   "embed_dim": 768},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: cross-model scaling")
    parser.add_argument("--model", default="vit_small_patch16_224",
                        choices=list(MODELS.keys()))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data/imagenet-1k")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip per-layer sensitivity analysis")
    args = parser.parse_args()

    device = get_device()
    model_info = MODELS[args.model]

    print("=" * 70)
    print(f"Phase 4 — {args.model}  ({model_info['params']} params)")
    print(f"PyTorch {torch.__version__}  |  device: {device}")
    print("=" * 70)

    results_dir = Path(f"results/Phase4/{args.model}")
    metrics_dir = results_dir / "metrics"
    plots_dir   = results_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load model + data config
    print("\nLoading model...")
    base_model = timm.create_model(args.model, pretrained=True, num_classes=1000)
    data_cfg   = timm.data.resolve_data_config({}, model=base_model)
    transform  = timm.data.create_transform(**data_cfg)
    print(f"  Data config: {data_cfg}")

    print("\nLoading ImageNet-1k validation...")
    loader = load_imagenet_val(transform, batch_size=args.batch_size,
                               data_dir=args.data_dir)

    results: dict = {
        "experiment":   "Phase4Scaling",
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch":      torch.__version__,
        "model":        args.model,
        "model_params": model_info["params"],
        "device":       str(device),
        "batch_size":   args.batch_size,
    }

    # ── FP32 baseline ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: FP32 baseline")
    print("=" * 70)
    model_fp32 = copy.deepcopy(base_model).to(device)
    mem_fp32 = model_size_mb(model_fp32)
    disk_fp32 = disk_size_mb(model_fp32)
    print(f"  Memory: {mem_fp32} MB")
    fp32_res = evaluate(model_fp32, loader, device, label="FP32")
    results["FP32"] = {**fp32_res, "memory_mb": mem_fp32, "disk_mb": disk_fp32}
    print(f"  Accuracy: {fp32_res['accuracy_percent']}%  |  "
          f"Latency: {fp32_res['avg_latency_ms_per_batch']} ms/batch")
    del model_fp32

    # ── FP16 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: FP16 (model.half())")
    print("=" * 70)
    model_fp16 = copy.deepcopy(base_model).half().to(device)
    mem_fp16 = model_size_mb(model_fp16)
    disk_fp16 = disk_size_mb(model_fp16)
    print(f"  Memory: {mem_fp16} MB  (reduction: "
          f"{mem_fp32/mem_fp16:.2f}x)")
    fp16_res = evaluate(model_fp16, loader, device, label="FP16")
    results["FP16"] = {**fp16_res, "memory_mb": mem_fp16, "disk_mb": disk_fp16}
    print(f"  Accuracy: {fp16_res['accuracy_percent']}%  |  "
          f"Latency: {fp16_res['avg_latency_ms_per_batch']} ms/batch")
    del model_fp16

    # ── INT8 per-tensor ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: INT8 per-tensor")
    print("=" * 70)
    model_int8pt = copy.deepcopy(base_model).to(device)
    model_int8pt, layer_stats = quantize_model_selective(model_int8pt, verbose=False)
    mem_int8pt = model_size_mb(model_int8pt)
    disk_int8pt = disk_size_mb(model_int8pt)
    print(f"  Memory: {mem_int8pt} MB  (reduction: "
          f"{mem_fp32/mem_int8pt:.2f}x)")
    int8pt_res = evaluate(model_int8pt, loader, device, label="INT8-pt")
    avg_mse = float(np.mean([s["mse"] for s in layer_stats]))
    worst   = max(layer_stats, key=lambda s: s["mse"])
    results["INT8-pt"] = {
        **int8pt_res, "memory_mb": mem_int8pt, "disk_mb": disk_int8pt,
        "n_layers_quant": len(layer_stats),
        "avg_mse": round(avg_mse, 10),
        "worst_layer": worst["layer"],
        "worst_layer_mse": round(worst["mse"], 10),
    }
    print(f"  Accuracy: {int8pt_res['accuracy_percent']}%  |  "
          f"Avg MSE: {avg_mse:.2e}")
    del model_int8pt

    # ── INT8 per-channel ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: INT8 per-channel")
    print("=" * 70)
    model_int8pc = copy.deepcopy(base_model).to(device)
    model_int8pc, _ = quantize_model_per_channel(model_int8pc, verbose=False)
    mem_int8pc = model_size_mb(model_int8pc)
    disk_int8pc = disk_size_mb(model_int8pc)
    int8pc_res = evaluate(model_int8pc, loader, device, label="INT8-pc")
    results["INT8-pc"] = {
        **int8pc_res, "memory_mb": mem_int8pc, "disk_mb": disk_int8pc,
    }
    print(f"  Accuracy: {int8pc_res['accuracy_percent']}%  |  "
          f"Memory: {mem_int8pc} MB")
    del model_int8pc

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Method':<12} {'Accuracy':>10} {'Δ FP32':>10} "
    header += f"{'Mem (MB)':>10} {'Disk (MB)':>10} {'Lat (ms)':>12}"
    print(header)
    print("-" * 70)
    fp32_acc = results["FP32"]["accuracy_percent"]
    for m in ["FP32", "FP16", "INT8-pt", "INT8-pc"]:
        r = results[m]
        delta = r["accuracy_percent"] - fp32_acc
        print(f"{m:<12} {r['accuracy_percent']:>9.3f}%  "
              f"{delta:>+9.4f}  {r['memory_mb']:>10.2f}  "
              f"{r['disk_mb']:>9.2f}  {r['avg_latency_ms_per_batch']:>11.1f}")

    # ── Save results ────────────────────────────────────────────────────
    out_path = metrics_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── Plots ───────────────────────────────────────────────────────────
    generate_plots(results, plots_dir)

    print("=" * 70)


if __name__ == "__main__":
    main()
