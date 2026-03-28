"""
Mini-experiment: INT8 weight-only quantization cu skip pe straturile de atenție.

Ipoteză (sugerată extern): cuantizând doar straturile MLP (fc1, fc2) și
lăsând straturile de atenție (attn.qkv, attn.proj) în FP32, acuratețea
poate fi mai bună decât cuantizarea completă.

Context din Faza 3 — sensitivity analysis:
  - Straturile de atenție sunt cele mai robuste individual (unele au
    degradare negativă, i.e. ușor mai bune după cuantizare)
  - Straturile MLP sunt cele mai sensibile individual
  → Teoretic, skipând attention (straturilerobuste) reducem compresia
    fără câștig clar de acuratețe față de per-channel complet.
  → Dar: sensitivity analysis cuantizează un strat izolat — interacțiunile
    la cuantizare simultană pot fi diferite. Merită verificat empiric.

Comparații:
  - FP32 baseline                       (referință)
  - INT8-pt complet   (48 straturi)     (Faza 2)
  - INT8-pc complet   (48 straturi)     (Faza 3)
  - INT8-mlp-only-pt  (24 straturi)     (acest experiment, per-tensor)
  - INT8-mlp-only-pc  (24 straturi)     (acest experiment, per-channel)

Output:
  results/experiments/mlp_only_int8/metrics/results.json
  results/experiments/mlp_only_int8/plots/

Usage:
  python scripts/experiments/mlp_only_int8.py
"""

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.quantized_linear import (
    SKIP_PATTERNS,
    quantize_model_selective,
    quantize_model_per_channel,
)

# ═══════════════════════════════════════════════════════════════════════════
# Skip patterns
# ═══════════════════════════════════════════════════════════════════════════

SKIP_FULL    = SKIP_PATTERNS                      # norm, cls_token, pos_embed, head
SKIP_MLP_ONLY = SKIP_PATTERNS + ["attn"]          # + attn.qkv, attn.proj

# ═══════════════════════════════════════════════════════════════════════════
# Results dirs
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results/experiments/mlp_only_int8")
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR   = RESULTS_DIR / "plots"

COLORS = {
    "FP32":             "#0072B2",
    "INT8-pt (full)":   "#E69F00",
    "INT8-pc (full)":   "#D55E00",
    "INT8-mlp-pt":      "#56B4E9",
    "INT8-mlp-pc":      "#009E73",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "legend.fontsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "#F8F8F8",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
})

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

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pin = torch.cuda.is_available()
    workers = 4 if torch.cuda.is_available() else 2

    return DataLoader(
        HFImageNet(hf_ds, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
    )

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
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


def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device, label: str = "",
             warmup: int = 3) -> dict:
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
        "accuracy_percent":          round(correct / total * 100, 3),
        "avg_loss":                  round(total_loss / len(loader), 6),
        "avg_latency_ms_per_batch":  round(float(np.mean(latencies)), 3),
        "total_samples":             total,
    }


def model_size_mb(model: nn.Module) -> float:
    total  = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total / 1024 ** 2, 3)


def count_quantized_layers(model: nn.Module) -> int:
    from src.models.quantized_linear import QuantizedLinear, QuantizedLinearPerChannel
    return sum(1 for m in model.modules()
               if isinstance(m, (QuantizedLinear, QuantizedLinearPerChannel)))

# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def generate_plots(results: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    fp32_acc = results["FP32"]["accuracy_percent"]

    accs   = [results[m]["accuracy_percent"] for m in methods]
    deltas = [acc - fp32_acc for acc in accs]
    mems   = [results[m]["memory_mb"] for m in methods]
    lats   = [results[m]["avg_latency_ms_per_batch"] for m in methods]
    n_layers = [results[m].get("n_layers_quant", 0) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MLP-only INT8 vs Full INT8 — ViT-Tiny / ImageNet-1k",
                 fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in zip(
        axes,
        [accs, mems, lats],
        ["Top-1 Accuracy (%)", "Memory (MB)", "Latency (ms/batch)"],
        ["Accuracy (%)", "MB", "ms/batch"],
    ):
        bars = ax.bar(methods, vals,
                      color=[COLORS.get(m, "#999") for m in methods], width=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "00_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Degradation bar
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, deltas,
                  color=[COLORS.get(m, "#999") for m in methods], width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Accuracy Degradation vs FP32 — MLP-only vs Full quantization")
    ax.set_ylabel("Δ Accuracy (pp)")
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    for i, (bar, val, n) in enumerate(zip(bars, deltas, n_layers)):
        label = f"{val:+.4f} pp"
        if n > 0:
            label += f"\n({n} layers)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.002 if val >= 0 else -0.006),
                label, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_degradation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {PLOTS_DIR}")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print("=" * 70)
    print("Experiment: MLP-only INT8 vs Full INT8 quantization")
    print(f"PyTorch {torch.__version__}  |  device: {device}")
    print("=" * 70)
    print()
    print("Skip patterns:")
    print(f"  Full quantization : {SKIP_FULL}")
    print(f"  MLP-only          : {SKIP_MLP_ONLY}")
    print()

    base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    data_cfg   = timm.data.resolve_data_config({}, model=base_model)
    transform  = timm.data.create_transform(**data_cfg)

    print("Loading ImageNet-1k validation...")
    loader = load_imagenet_val(transform, batch_size=64)

    results: dict = {}

    configs = [
        ("FP32",           lambda m: m,                                                        "FP32 baseline"),
        ("INT8-pt (full)", lambda m: quantize_model_selective(m, skip_patterns=SKIP_FULL,    verbose=False)[0], "INT8 per-tensor, 48 straturi"),
        ("INT8-pc (full)", lambda m: quantize_model_per_channel(m, skip_patterns=SKIP_FULL,  verbose=False)[0], "INT8 per-channel, 48 straturi"),
        ("INT8-mlp-pt",    lambda m: quantize_model_selective(m, skip_patterns=SKIP_MLP_ONLY, verbose=False)[0], "INT8 per-tensor, MLP only"),
        ("INT8-mlp-pc",    lambda m: quantize_model_per_channel(m, skip_patterns=SKIP_MLP_ONLY, verbose=False)[0], "INT8 per-channel, MLP only"),
    ]

    for label, quantize_fn, desc in configs:
        print(f"\n{'='*70}")
        print(f"{label}: {desc}")
        print("=" * 70)
        model = copy.deepcopy(base_model).to(device)
        model = quantize_fn(model)
        n_q   = count_quantized_layers(model)
        mem   = model_size_mb(model)
        print(f"  Quantized layers: {n_q}  |  Memory: {mem} MB")
        res = evaluate(model, loader, device, label=label)
        results[label] = {**res, "memory_mb": mem, "n_layers_quant": n_q}
        print(f"  Accuracy: {res['accuracy_percent']}%")
        del model

    # Summary
    fp32_acc = results["FP32"]["accuracy_percent"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':>10} {'Δ FP32':>10} "
          f"{'Layers':>8} {'Mem (MB)':>10} {'Lat (ms)':>12}")
    print("-" * 74)
    for label, _, _ in configs:
        r = results[label]
        delta = r["accuracy_percent"] - fp32_acc
        print(f"{label:<20} {r['accuracy_percent']:>9.3f}%  "
              f"{delta:>+9.4f}  {r['n_layers_quant']:>7}  "
              f"{r['memory_mb']:>9.2f}  "
              f"{r['avg_latency_ms_per_batch']:>11.1f}")

    # Verdict
    full_pt_acc  = results["INT8-pt (full)"]["accuracy_percent"]
    mlp_pt_acc   = results["INT8-mlp-pt"]["accuracy_percent"]
    full_pc_acc  = results["INT8-pc (full)"]["accuracy_percent"]
    mlp_pc_acc   = results["INT8-mlp-pc"]["accuracy_percent"]

    print("\nVERDICT:")
    print(f"  MLP-only pt vs Full pt:  {mlp_pt_acc - full_pt_acc:+.4f} pp "
          f"({'mai bun' if mlp_pt_acc > full_pt_acc else 'mai slab'})")
    print(f"  MLP-only pc vs Full pc:  {mlp_pc_acc - full_pc_acc:+.4f} pp "
          f"({'mai bun' if mlp_pc_acc > full_pc_acc else 'mai slab'})")
    print(f"  MLP-only pt vs INT8-pc full: {mlp_pt_acc - full_pc_acc:+.4f} pp")

    # Save
    output = {
        "experiment":   "mlp_only_int8",
        "hypothesis":   "Skipping attention layers during INT8 quantization improves accuracy",
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":        "vit_tiny_patch16_224",
        "dataset":      "ImageNet-1k validation (50000 images)",
        "device":       str(device),
        "skip_patterns": {
            "full":     SKIP_FULL,
            "mlp_only": SKIP_MLP_ONLY,
        },
        "results":      results,
        "verdict": {
            "mlp_only_pt_vs_full_pt_pp": round(mlp_pt_acc - full_pt_acc, 4),
            "mlp_only_pc_vs_full_pc_pp": round(mlp_pc_acc - full_pc_acc, 4),
            "hypothesis_confirmed":      mlp_pt_acc > full_pt_acc or mlp_pc_acc > full_pc_acc,
        },
    }
    out_path = METRICS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")

    generate_plots(results)
    print("=" * 70)


if __name__ == "__main__":
    main()
