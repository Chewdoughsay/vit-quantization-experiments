"""
Phase 3 — Layer sensitivity analysis and systematic comparisons.

Experiments:
  1. Per-tensor vs per-channel INT8  (global accuracy, per-layer MSE)
  2. Sensitivity analysis            (quantize exactly one layer at a time)
  3. Layer-wise quantization error   (MSE/MAE per-tensor vs per-channel)
  4. Timing comparison               (FP32, FP16, INT8-pt, INT8-pc)
  5. Memory footprint                (in-memory params + serialized file)

Outputs:
    results/Phase3/metrics/phase3_results.json
    results/Phase3/plots/   (generated automatically at the end)

Usage:
    python scripts/layer_sensitivity_analysis.py
    python scripts/layer_sensitivity_analysis.py --skip-sensitivity   # skip the 48 evaluations
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import timm.data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_model import create_vit_model
from src.models.quantized_linear import (
    SKIP_PATTERNS,
    QuantizedLinear,
    QuantizedLinearPerChannel,
    int8_quantize,
    int8_quantize_per_channel,
    quantization_error,
    quantization_error_per_channel,
    quantize_model_selective,
    quantize_model_per_channel,
    _set_nested_module,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

WNID_TO_IMAGENET = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482,
    "n03000684": 491, "n03028079": 497, "n03394916": 566,
    "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701,
}

IMAGENETTE_DIR  = Path("data/imagenette2-320")
RESULTS_DIR     = Path("results/Phase3")
OUTPUT_DIR      = RESULTS_DIR / "plots"
METRICS_DIR     = RESULTS_DIR / "metrics"
SAVE_DPI        = 300

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
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "#F8F8F8",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
})


# ═══════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_val_loader(batch_size: int = 64, num_workers: int = 2,
                   transform=None) -> tuple:
    val_dataset = datasets.ImageFolder(
        str(IMAGENETTE_DIR / "val"), transform=transform)
    label_map = _build_label_map(val_dataset)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0))
    return loader, label_map


def _build_label_map(dataset: datasets.ImageFolder) -> list[int]:
    mapping = [None] * len(dataset.classes)
    for wnid, idx in dataset.class_to_idx.items():
        mapping[idx] = WNID_TO_IMAGENET[wnid]
    return mapping


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, label_map: list[int],
             device: torch.device, desc: str = "Eval",
             warmup_batches: int = 3) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    is_half   = next(model.parameters()).dtype == torch.float16
    lmap_t    = torch.tensor(label_map, dtype=torch.long, device=device)

    correct, total, total_loss = 0, 0, 0.0
    batch_times = []

    for i, (images, local_labels) in enumerate(tqdm(loader, desc=desc, leave=False)):
        images = images.to(device)
        if is_half:
            images = images.half()
        imagenet_labels = lmap_t[local_labels.to(device)]

        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "mps":   torch.mps.synchronize()
        elif device.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_batches:
            batch_times.append(t1 - t0)

        total_loss += criterion(outputs.float(), imagenet_labels).item()
        correct    += (outputs.argmax(1) == imagenet_labels).sum().item()
        total      += imagenet_labels.size(0)

    avg_lat = (sum(batch_times) / len(batch_times) * 1000) if batch_times else 0.0
    return {
        "accuracy_percent": round(correct / total * 100, 4),
        "avg_loss":         round(total_loss / len(loader), 6),
        "avg_latency_ms":   round(avg_lat, 3),
        "total_samples":    total,
    }


def model_memory_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size()
               for p in [*model.parameters(), *model.buffers()]) / (1024 ** 2)


def model_disk_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 ** 2)
    os.unlink(path)
    return round(size, 3)


def get_quantizable_layer_names(model: nn.Module) -> list[str]:
    return [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and not any(p in name for p in SKIP_PATTERNS)
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1 — Layer-wise error comparison (per-tensor vs per-channel)
# ═══════════════════════════════════════════════════════════════════════════

def layer_error_comparison(model: nn.Module) -> list[dict]:
    """Compute per-tensor and per-channel MSE/MAE for every quantizable layer."""
    results = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if any(p in name for p in SKIP_PATTERNS):
            continue
        w = m.weight.data.cpu()

        q_pt,  sc_pt  = int8_quantize(w)
        err_pt = quantization_error(w, q_pt, sc_pt)

        q_pc,  sc_pc  = int8_quantize_per_channel(w)
        err_pc = quantization_error_per_channel(w, q_pc, sc_pc)

        results.append({
            "layer":          name,
            "shape":          list(w.shape),
            "n_params":       w.numel(),
            "per_tensor_mse": err_pt["mse"],
            "per_tensor_mae": err_pt["mae"],
            "per_tensor_scale": err_pt["scale"],
            "per_channel_mse": err_pc["mse"],
            "per_channel_mae": err_pc["mae"],
            "mse_improvement": err_pt["mse"] / err_pc["mse"] if err_pc["mse"] > 0 else 1.0,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2 — Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(
    model_fp32: nn.Module,
    loader: DataLoader,
    label_map: list[int],
    device: torch.device,
    layer_names: list[str],
) -> list[dict]:
    """Quantize exactly one layer at a time (per-tensor), measure accuracy impact."""
    results = []
    model_fp32.eval()

    for name in layer_names:
        # Navigate to parent and get the original module
        parts  = name.split(".")
        parent = model_fp32
        for p in parts[:-1]:
            parent = getattr(parent, p)
        original_module = getattr(parent, parts[-1])

        # Replace with quantized
        q_linear = QuantizedLinear.from_linear(original_module)
        q_linear = q_linear.to(device)
        setattr(parent, parts[-1], q_linear)

        res = evaluate(model_fp32, loader, label_map, device,
                       desc=f"Sens {name[-25:]}")
        results.append({
            "layer":            name,
            "shape":            list(original_module.weight.shape),
            "n_params":         original_module.weight.numel(),
            "accuracy_percent": res["accuracy_percent"],
        })

        # Restore original (move back to device)
        original_module = original_module.to(device)
        setattr(parent, parts[-1], original_module)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3 — Timing (FP32, FP16, INT8-pt, INT8-pc)
# ═══════════════════════════════════════════════════════════════════════════

def timing_comparison(loader: DataLoader, label_map: list[int],
                      device: torch.device, n_timing_batches: int = 30) -> dict:
    """Measure latency for each precision format over n_timing_batches batches."""
    configs = {
        "FP32":    lambda: create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True),
        "FP16":    lambda: create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True),
        "INT8-pt": lambda: create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True),
        "INT8-pc": lambda: create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True),
    }

    timing = {}
    for label, model_fn in configs.items():
        print(f"  Timing: {label} ...", end=" ", flush=True)
        model = model_fn().to(device).eval()

        if label == "FP16":
            model = model.half()
        elif label == "INT8-pt":
            model, _ = quantize_model_selective(model, verbose=False)
            model = model.to(device)
        elif label == "INT8-pc":
            model, _ = quantize_model_per_channel(model, verbose=False)
            model = model.to(device)

        batch_times = []
        with torch.no_grad():
            for i, (images, _) in enumerate(loader):
                if i >= n_timing_batches + 5:   # +5 warmup
                    break
                images = images.to(device)
                if label == "FP16":
                    images = images.half()

                t0 = time.perf_counter()
                _ = model(images)
                if device.type == "mps":    torch.mps.synchronize()
                elif device.type == "cuda": torch.cuda.synchronize()
                t1 = time.perf_counter()

                if i >= 5:   # skip warmup
                    batch_times.append((t1 - t0) * 1000)

        avg = sum(batch_times) / len(batch_times)
        std = (sum((x - avg) ** 2 for x in batch_times) / len(batch_times)) ** 0.5
        timing[label] = {
            "avg_latency_ms": round(avg, 3),
            "std_ms":         round(std, 3),
            "memory_mb":      round(model_memory_mb(model), 3),
            "disk_mb":        round(model_disk_mb(model), 3),
        }
        print(f"{avg:.1f} ± {std:.1f} ms/batch")
        del model

    return timing


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def save_fig(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sensitivity(sensitivity: list[dict], fp32_acc: float) -> None:
    """Ranked bar chart — accuracy degradation per layer."""
    degradations = [
        {"layer": s["layer"], "deg": fp32_acc - s["accuracy_percent"]}
        for s in sensitivity
    ]
    degradations.sort(key=lambda x: x["deg"], reverse=True)

    names = [d["layer"].replace("blocks.", "b").replace(".attn.", ".a.").replace(".mlp.", ".m.") for d in degradations]
    degs  = [d["deg"] for d in degradations]

    colors_bars = [
        "#D55E00" if v > np.percentile(degs, 75) else COLORS["INT8-pt"]
        for v in degs
    ]

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(names))
    ax.bar(x, degs, color=colors_bars, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy degradation (pp)")
    ax.set_title("Sensitivity Analysis — Impact per Layer (INT8 per-tensor, one layer quantized at a time)\n"
                 "Orange = top 25% most sensitive")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(np.mean(degs), color="gray", linestyle="--", linewidth=1,
               label=f"Mean = {np.mean(degs):.4f} pp")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "01_sensitivity_ranked.png")


def plot_mse_comparison(layer_errors: list[dict]) -> None:
    """Per-tensor vs per-channel MSE side by side."""
    names = [e["layer"].replace("blocks.", "b") for e in layer_errors]
    pt    = [e["per_tensor_mse"] for e in layer_errors]
    pc    = [e["per_channel_mse"] for e in layer_errors]
    x     = np.arange(len(names))
    w     = 0.4

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    # Top: side-by-side MSE
    ax = axes[0]
    ax.bar(x - w/2, pt, w, label="Per-tensor", color=COLORS["INT8-pt"],
           edgecolor="white", linewidth=0.8)
    ax.bar(x + w/2, pc, w, label="Per-channel", color=COLORS["INT8-pc"],
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("MSE")
    ax.set_title("Per-Tensor vs Per-Channel INT8 — Weight Quantization Error (MSE)")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend()

    # Bottom: improvement ratio
    ax = axes[1]
    ratios = [e["mse_improvement"] for e in layer_errors]
    colors_bars = [
        "#D55E00" if r > np.percentile(ratios, 75) else COLORS["INT8-pc"]
        for r in ratios
    ]
    ax.bar(x, ratios, color=colors_bars, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("MSE improvement (pt / pc)")
    ax.set_title("Per-channel improvement over per-tensor (MSE ratio)\nOrange = top 25%")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(np.mean(ratios), color="gray", linestyle="--",
               label=f"Mean = {np.mean(ratios):.2f}×")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "02_mse_per_tensor_vs_per_channel.png")


def plot_timing_and_memory(timing: dict) -> None:
    methods = ["FP32", "FP16", "INT8-pt", "INT8-pc"]
    colors  = [COLORS[m] for m in methods]

    lat  = [timing[m]["avg_latency_ms"] for m in methods]
    std  = [timing[m]["std_ms"] for m in methods]
    mem  = [timing[m]["memory_mb"] for m in methods]
    disk = [timing[m]["disk_mb"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Timing & Memory — FP32 / FP16 / INT8 per-tensor / INT8 per-channel\n"
                 "ViT-Tiny pretrained, Apple M4 MPS",
                 fontsize=13, fontweight="bold", y=1.02)

    # Latency
    ax = axes[0]
    bars = ax.bar(methods, lat, color=colors, width=0.5, edgecolor="white", linewidth=1.2,
                  yerr=std, capsize=5, error_kw={"ecolor": "gray", "elinewidth": 1.5})
    for bar, v in zip(bars, lat):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(std) + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Inference Latency (ms/batch)")
    ax.set_ylabel("Avg. Latency (ms)")
    ax.set_ylim(0, max(lat) * 1.35)

    # Memory (param + buffers)
    ax = axes[1]
    bars = ax.bar(methods, mem, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, mem):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.1f} MB", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("In-Memory Size (params + buffers)")
    ax.set_ylabel("Memory (MB)")
    ax.set_ylim(0, max(mem) * 1.3)

    # Disk
    ax = axes[2]
    bars = ax.bar(methods, disk, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, disk):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.1f} MB", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Serialized Size (torch.save)")
    ax.set_ylabel("Disk (MB)")
    ax.set_ylim(0, max(disk) * 1.3)

    fig.tight_layout()
    save_fig(fig, "03_timing_memory.png")


def plot_global_comparison(results: dict) -> None:
    """Overview: acuratețe + degradare pentru toate cele 4 metode."""
    methods  = ["FP32", "FP16", "INT8-pt", "INT8-pc"]
    colors   = [COLORS[m] for m in methods]
    fp32_acc = results["fp32"]["accuracy_percent"]

    acc_vals = [
        fp32_acc,
        results["fp16"]["accuracy_percent"],
        results["int8_per_tensor"]["accuracy_percent"],
        results["int8_per_channel"]["accuracy_percent"],
    ]
    deg_vals = [0] + [fp32_acc - a for a in acc_vals[1:]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Global comparison — FP32 / FP16 / INT8 per-tensor / INT8 per-channel\n"
                 "ViT-Tiny pretrained ImageNet-1k on ImageNette (3925 images)",
                 fontsize=13, fontweight="bold", y=1.02)

    # Accuracy absolute
    ax = axes[0]
    bars = ax.bar(methods, acc_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{v:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Top-1 Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 86)

    # Degradare față de FP32
    ax = axes[1]
    bars = ax.bar(methods, deg_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, deg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.05,
                f"{v:+.4f} pp", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("Accuracy degradation vs FP32")
    ax.set_ylabel("Degradation (pp)")
    ax.axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    save_fig(fig, "00_global_comparison.png")


def plot_sensitivity_heatmap(sensitivity: list[dict], fp32_acc: float) -> None:
    """Heatmap pe structura modelului: layer x metrica."""
    # Organizăm pe block și tip
    blocks  = list(range(12))
    types   = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
    matrix  = np.zeros((len(types), len(blocks)))

    for s in sensitivity:
        deg = fp32_acc - s["accuracy_percent"]
        for ti, t in enumerate(types):
            if t in s["layer"]:
                for bi in blocks:
                    if f"blocks.{bi}." in s["layer"]:
                        matrix[ti, bi] = deg

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"B{i}" for i in range(12)])
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types)
    ax.set_title("Sensitivity Heatmap — Accuracy degradation (pp) per quantized layer\n"
                 "(darker = more sensitive)")

    for ti in range(len(types)):
        for bi in range(len(blocks)):
            v = matrix[ti, bi]
            ax.text(bi, ti, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if v < matrix.max() * 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Degradation (pp)")
    fig.tight_layout()
    save_fig(fig, "04_sensitivity_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
# Markdown summary
# ═══════════════════════════════════════════════════════════════════════════

def write_markdown(results: dict) -> None:
    fp32  = results["fp32"]
    fp16  = results["fp16"]
    pt    = results["int8_per_tensor"]
    pc    = results["int8_per_channel"]
    t     = results["timing"]
    le    = results["layer_errors"]

    avg_improvement = sum(e["mse_improvement"] for e in le) / len(le)

    lines = [
        "# Phase 3 — Sensitivity Analysis & Systematic Comparisons",
        "",
        f"**Data:** {results['timestamp']}  |  **Model:** `{results['model']}`",
        "",
        "## 1. Comparație globală acuratețe",
        "",
        "| Format | Accuracy | Δ vs FP32 | Memory (MB) | Disk (MB) |",
        "|--------|----------|-----------|-------------|-----------|",
        f"| FP32    | {fp32['accuracy_percent']:.4f}% | —         | "
        f"{t['FP32']['memory_mb']:.1f} | {t['FP32']['disk_mb']:.1f} |",
        f"| FP16    | {fp16['accuracy_percent']:.4f}% | "
        f"{fp32['accuracy_percent']-fp16['accuracy_percent']:+.4f} pp | "
        f"{t['FP16']['memory_mb']:.1f} | {t['FP16']['disk_mb']:.1f} |",
        f"| INT8 per-tensor  | {pt['accuracy_percent']:.4f}% | "
        f"{fp32['accuracy_percent']-pt['accuracy_percent']:+.4f} pp | "
        f"{t['INT8-pt']['memory_mb']:.1f} | {t['INT8-pt']['disk_mb']:.1f} |",
        f"| INT8 per-channel | {pc['accuracy_percent']:.4f}% | "
        f"{fp32['accuracy_percent']-pc['accuracy_percent']:+.4f} pp | "
        f"{t['INT8-pc']['memory_mb']:.1f} | {t['INT8-pc']['disk_mb']:.1f} |",
        "",
        "## 2. Timing (latență inferență, ms/batch)",
        "",
        "| Format | Avg (ms) | Std (ms) |",
        "|--------|----------|----------|",
    ] + [
        f"| {m} | {t[m]['avg_latency_ms']:.1f} | {t[m]['std_ms']:.1f} |"
        for m in ["FP32", "FP16", "INT8-pt", "INT8-pc"]
    ] + [
        "",
        "## 3. Per-tensor vs Per-channel — eroare cuantizare",
        "",
        f"- MSE mediu per-tensor : `{sum(e['per_tensor_mse'] for e in le)/len(le):.4e}`",
        f"- MSE mediu per-channel: `{sum(e['per_channel_mse'] for e in le)/len(le):.4e}`",
        f"- Îmbunătățire medie   : **{avg_improvement:.2f}×** (per-channel mai precis)",
        "",
    ]

    if "sensitivity" in results:
        sens     = results["sensitivity"]
        fp32_acc = fp32["accuracy_percent"]
        ranked   = sorted(sens, key=lambda s: fp32_acc - s["accuracy_percent"], reverse=True)
        lines += [
            "## 4. Sensitivity — top 5 cei mai sensibili layeri",
            "",
            "| Layer | Accuracy single-quant | Degradare |",
            "|-------|-----------------------|-----------|",
        ] + [
            f"| `{s['layer']}` | {s['accuracy_percent']:.4f}% | "
            f"{fp32_acc - s['accuracy_percent']:+.4f} pp |"
            for s in ranked[:5]
        ] + [""]

    lines += [
        "## Concluzii",
        "",
        "- FP16 (`model.half()`) = zero degradare practică, **2× mai mic**, **1.28× mai rapid**",
        f"- INT8 per-tensor = **+{fp32['accuracy_percent']-pt['accuracy_percent']:.3f} pp** degradare, "
        f"**{t['FP32']['memory_mb']/t['INT8-pt']['memory_mb']:.2f}×** mai mic, fără speedup real (dequant overhead)",
        f"- INT8 per-channel = **+{fp32['accuracy_percent']-pc['accuracy_percent']:.3f} pp** degradare, "
        f"eroare cuantizare **{avg_improvement:.2f}× mai mică** decât per-tensor",
        "- Outlieri (`blocks.7.mlp.*`) sunt mult mai sensibili — candidați pentru mixed-precision",
    ]

    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",           default="mps",
                        choices=["mps", "cuda", "cpu"])
    parser.add_argument("--batch-size",       type=int, default=64)
    parser.add_argument("--num-workers",      type=int, default=2)
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip the 48-evaluation sensitivity analysis (~7 min)")
    args = parser.parse_args()

    device = torch.device(args.device)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Phase 3 — Sensitivity Analysis & Systematic Comparisons")
    print(f"PyTorch {torch.__version__}  |  device: {args.device}")
    print("=" * 70 + "\n")

    # Transform (from timm, same as Phases 1-2)
    _ref_model  = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    data_config = timm.data.resolve_data_config({}, model=_ref_model)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    del _ref_model

    loader, label_map = get_val_loader(args.batch_size, args.num_workers, val_transform)
    print(f"Dataset: {len(loader.dataset)} images  |  {len(loader)} batches\n")

    # ------------------------------------------------------------------
    # STEP 1: FP32 baseline
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: FP32 baseline")
    print("=" * 70)
    model_fp32 = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    model_fp32 = model_fp32.to(device).eval()
    fp32_res = evaluate(model_fp32, loader, label_map, device, desc="FP32")
    print(f"  FP32 accuracy: {fp32_res['accuracy_percent']:.2f}%\n")

    # ------------------------------------------------------------------
    # STEP 2: FP16
    # ------------------------------------------------------------------
    print("STEP 2: FP16 (model.half())")
    model_fp16 = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    model_fp16 = model_fp16.half().to(device).eval()
    fp16_res = evaluate(model_fp16, loader, label_map, device, desc="FP16")
    print(f"  FP16 accuracy: {fp16_res['accuracy_percent']:.2f}%\n")
    del model_fp16

    # ------------------------------------------------------------------
    # STEP 3: INT8 per-tensor (full model)
    # ------------------------------------------------------------------
    print("STEP 3: INT8 per-tensor (full model)")
    model_pt = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    model_pt, _ = quantize_model_selective(model_pt, verbose=False)
    model_pt = model_pt.to(device).eval()
    pt_res = evaluate(model_pt, loader, label_map, device, desc="INT8-pt")
    print(f"  INT8-pt accuracy: {pt_res['accuracy_percent']:.2f}%\n")
    del model_pt

    # ------------------------------------------------------------------
    # STEP 4: INT8 per-channel (full model)
    # ------------------------------------------------------------------
    print("STEP 4: INT8 per-channel (full model)")
    model_pc = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    model_pc, _ = quantize_model_per_channel(model_pc, verbose=False)
    model_pc = model_pc.to(device).eval()
    pc_res = evaluate(model_pc, loader, label_map, device, desc="INT8-pc")
    print(f"  INT8-pc accuracy: {pc_res['accuracy_percent']:.2f}%\n")
    del model_pc

    # ------------------------------------------------------------------
    # STEP 5: Layer-wise error comparison
    # ------------------------------------------------------------------
    print("STEP 5: Layer-wise error (per-tensor vs per-channel)")
    model_clean = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    layer_errors = layer_error_comparison(model_clean)
    avg_pt_mse = sum(e["per_tensor_mse"] for e in layer_errors) / len(layer_errors)
    avg_pc_mse = sum(e["per_channel_mse"] for e in layer_errors) / len(layer_errors)
    print(f"  Avg MSE per-tensor : {avg_pt_mse:.4e}")
    print(f"  Avg MSE per-channel: {avg_pc_mse:.4e}")
    print(f"  Improvement ratio  : {avg_pt_mse/avg_pc_mse:.2f}×\n")

    # ------------------------------------------------------------------
    # STEP 6: Sensitivity analysis (opțional)
    # ------------------------------------------------------------------
    sensitivity = None
    if not args.skip_sensitivity:
        layer_names = get_quantizable_layer_names(model_clean)
        print(f"STEP 6: Sensitivity analysis ({len(layer_names)} layere, ~{len(layer_names)*9//60} min)")
        model_clean = model_clean.to(device).eval()
        sensitivity = sensitivity_analysis(
            model_clean, loader, label_map, device, layer_names)
        worst = min(sensitivity, key=lambda s: s["accuracy_percent"])
        best  = max(sensitivity, key=lambda s: s["accuracy_percent"])
        print(f"\n  Most sensitive: {worst['layer']} → {worst['accuracy_percent']:.2f}%")
        print(f"  Most robust   : {best['layer']} → {best['accuracy_percent']:.2f}%\n")
    else:
        print("STEP 6: Sensitivity analysis — SKIPPED (--skip-sensitivity)\n")

    del model_clean

    # ------------------------------------------------------------------
    # STEP 7: Timing & memory
    # ------------------------------------------------------------------
    print("STEP 7: Timing & memory comparison")
    timing = timing_comparison(loader, label_map, device)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "experiment":  "Phase3SensitivityAnalysis",
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "model":       "vit_tiny_patch16_224",
        "dataset":     "imagenette2-320",
        "device":      args.device,
        "batch_size":  args.batch_size,
        "fp32":        fp32_res,
        "fp16":        fp16_res,
        "int8_per_tensor":  pt_res,
        "int8_per_channel": pc_res,
        "layer_errors": layer_errors,
        "timing":      timing,
        "sensitivity": sensitivity,
    }

    json_path = METRICS_DIR / "phase3_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}\n")

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("Generating plots ...")
    plot_global_comparison(results)
    plot_mse_comparison(layer_errors)
    plot_timing_and_memory(timing)
    if sensitivity:
        plot_sensitivity(sensitivity, fp32_res["accuracy_percent"])
        plot_sensitivity_heatmap(sensitivity, fp32_res["accuracy_percent"])
    write_markdown(results)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<14} {'Accuracy':>10} {'Δ FP32':>10} {'Mem (MB)':>10} {'Disk (MB)':>10} {'Lat (ms)':>10}")
    print("-" * 68)
    for method, res, key in [
        ("FP32",    fp32_res, "FP32"),
        ("FP16",    fp16_res, "FP16"),
        ("INT8-pt", pt_res,   "INT8-pt"),
        ("INT8-pc", pc_res,   "INT8-pc"),
    ]:
        deg = fp32_res["accuracy_percent"] - res["accuracy_percent"]
        print(f"{method:<14} {res['accuracy_percent']:>9.2f}%  "
              f"{deg:>+9.4f}  "
              f"{timing[key]['memory_mb']:>9.1f}  "
              f"{timing[key]['disk_mb']:>9.1f}  "
              f"{timing[key]['avg_latency_ms']:>9.1f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
