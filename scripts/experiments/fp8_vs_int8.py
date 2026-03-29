"""
Experiment: FP8 E4M3FN vs INT8 weight-only quantization.

Replică structura din Faza 2 (QuantizedLinear cu stocare reală, cuantizare
selectivă) dar folosind FP8 E4M3FN în loc de INT8. Compară direct:

  FP32       — baseline
  INT8-pt    — Faza 2 (scalare per-tensor, 8 biți întregi)
  INT8-pc    — Faza 3 (scalare per-channel, 8 biți întregi)
  FP8-pt     — FP8 E4M3FN per-tensor  (acest experiment)
  FP8-pc     — FP8 E4M3FN per-channel (acest experiment)

Diferența față de Faza 0 (preliminar):
  - Faza 0: cast FP8 și înapoi imediat, stocare în FP32, fără reducere memorie
  - Acest script: ponderile rămân în float8_e4m3fn (stocare reală),
    dequantizare abia la forward pass — aceeași abordare ca INT8 din Faza 2.

Notă hardware: pe Apple M4 MPS, float8_e4m3fn nu are kernele native —
operațiile sunt simulate pe CPU. Speedup-ul real se vede pe A100/H100.
Acuratețea este corectă pe orice hardware.

Output:
    results/experiments/fp8_vs_int8/metrics/results.json
    results/experiments/fp8_vs_int8/plots/

Usage:
    python scripts/experiments/fp8_vs_int8.py
"""

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
    SKIP_PATTERNS,
    quantize_model_selective,
    quantize_model_per_channel,
)

# ═══════════════════════════════════════════════════════════════════════════
# FP8 Linear — stocare reală în float8_e4m3fn
# ═══════════════════════════════════════════════════════════════════════════

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX   = torch.finfo(FP8_DTYPE).max  # 448.0


class FP8Linear(nn.Module):
    """Drop-in nn.Linear replacement cu ponderi stocate în float8_e4m3fn.

    La forward: dequantizează ponderile la float32, face matmul normal.
    Același pattern ca QuantizedLinear din Faza 2, dar cu FP8 în loc de INT8.
    """

    def __init__(self, q_weight: torch.Tensor, scale, bias=None,
                 per_channel: bool = False):
        super().__init__()
        # Stocăm în float8 pe CPU (MPS nu suportă float8 buffers)
        self.register_buffer("q_weight", q_weight.cpu())
        self.register_buffer("scale", scale.cpu() if isinstance(scale, torch.Tensor)
                             else torch.tensor(scale))
        self.bias        = nn.Parameter(bias.clone()) if bias is not None else None
        self.per_channel = per_channel

    @classmethod
    def from_linear(cls, module: nn.Linear, per_channel: bool = False):
        w = module.weight.data.float().cpu()
        if per_channel:
            scales = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
            scale  = FP8_MAX / scales
            q      = (w * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            scale_store = (1.0 / scale)  # store 1/scale for dequant
        else:
            abs_max = w.abs().max().clamp(min=1e-12)
            scale   = FP8_MAX / abs_max
            q       = (w * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            scale_store = torch.tensor(1.0 / scale.item())

        return cls(q, scale_store, module.bias, per_channel=per_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on CPU (float8 not supported on MPS), then move to device
        w = self.q_weight.to(torch.float32)
        if self.per_channel:
            w = w * self.scale          # both on CPU
        else:
            w = w * self.scale.item()   # scalar, no device issue
        w = w.to(x.device)
        bias = self.bias.to(x.device) if self.bias is not None else None
        return nn.functional.linear(x, w, bias)

    def extra_repr(self) -> str:
        gran = "per-channel" if self.per_channel else "per-tensor"
        return (f"out={self.q_weight.shape[0]}, in={self.q_weight.shape[1]}, "
                f"fp8={FP8_DTYPE}, {gran}")


def _set_nested(model: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def quantize_model_fp8(
    model: nn.Module,
    per_channel: bool = False,
    skip_patterns=None,
    verbose: bool = False,
) -> tuple[nn.Module, list[dict]]:
    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS

    layer_stats = []
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear)]

    for name, module in targets:
        if any(p in name for p in skip_patterns):
            continue
        # FP8 cast must happen on CPU — MPS does not support float8_e4m3fn
        w = module.weight.data.float().cpu()

        if per_channel:
            scales = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
            w_q    = (w * FP8_MAX / scales).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float()
            w_q    = w_q * scales / FP8_MAX
        else:
            abs_max = w.abs().max().clamp(min=1e-12)
            w_q     = (w * FP8_MAX / abs_max).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float()
            w_q     = w_q * abs_max / FP8_MAX

        mse = float(((w - w_q) ** 2).mean())
        layer_stats.append({"layer": name, "mse": mse,
                            "shape": list(w.shape)})

        _set_nested(model, name, FP8Linear.from_linear(module, per_channel=per_channel))

    if verbose:
        print(f"  FP8 quantized: {len(layer_stats)} layers")
        for s in layer_stats:
            print(f"    {s['layer']:<50} MSE={s['mse']:.2e}")

    return model, layer_stats


# ═══════════════════════════════════════════════════════════════════════════
# Device / DataLoader / Eval
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


class HFImageNet(Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img  = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), item["label"]


def load_imagenet_val(transform, batch_size=64,
                      data_dir="data/imagenet-1k") -> DataLoader:
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

    pin     = torch.cuda.is_available()
    workers = 4 if torch.cuda.is_available() else 2
    return DataLoader(HFImageNet(hf_ds, transform), batch_size=batch_size,
                      shuffle=False, num_workers=workers, pin_memory=pin,
                      persistent_workers=(workers > 0))


def evaluate(model, loader, device, label="", warmup=3) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    latencies = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc=label, leave=True)):
            images, labels = images.to(device), labels.to(device)
            sync_device(device)
            t0 = time.perf_counter()
            outputs = model(images)
            sync_device(device)
            t1 = time.perf_counter()
            loss = criterion(outputs, labels)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            total_loss += loss.item()
            if i >= warmup:
                latencies.append((t1 - t0) * 1000)

    return {
        "accuracy_percent":         round(correct / total * 100, 3),
        "avg_loss":                  round(total_loss / len(loader), 6),
        "avg_latency_ms_per_batch":  round(float(np.mean(latencies)), 3),
        "total_samples":             total,
    }


def model_size_mb(model) -> float:
    total  = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total / 1024 ** 2, 3)


def disk_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return round(Path(f.name).stat().st_size / 1024 ** 2, 3)


def count_quant_layers(model) -> int:
    from src.models.quantized_linear import QuantizedLinear, QuantizedLinearPerChannel
    n = sum(1 for m in model.modules()
            if isinstance(m, (QuantizedLinear, QuantizedLinearPerChannel, FP8Linear)))
    return n

# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

PLOTS_DIR = Path("results/experiments/fp8_vs_int8/plots")

COLORS = {
    "FP32":    "#0072B2",
    "INT8-pt": "#E69F00",
    "INT8-pc": "#D55E00",
    "FP8-pt":  "#CC79A7",
    "FP8-pc":  "#009E73",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 13, "legend.fontsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "#F8F8F8",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
})


def generate_plots(results: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    methods  = list(results.keys())
    fp32_acc = results["FP32"]["accuracy_percent"]

    accs   = [results[m]["accuracy_percent"] for m in methods]
    deltas = [results[m]["accuracy_percent"] - fp32_acc for m in methods]
    mems   = [results[m]["memory_mb"] for m in methods]
    lats   = [results[m]["avg_latency_ms_per_batch"] for m in methods]

    # ── 00 Summary ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FP8 E4M3FN vs INT8 — ViT-Tiny / ImageNet-1k validation",
                 fontsize=14, fontweight="bold")
    for ax, vals, title, ylabel in zip(
        axes,
        [accs, mems, lats],
        ["Top-1 Accuracy (%)", "Memory (MB)", "Latency (ms/batch)"],
        ["Accuracy (%)", "MB", "ms/batch"],
    ):
        bars = ax.bar(methods, vals,
                      color=[COLORS.get(m, "#999") for m in methods],
                      width=0.55, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "00_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 01 Degradation ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, deltas,
                  color=[COLORS.get(m, "#999") for m in methods],
                  width=0.55, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_title("Degradare acuratețe față de FP32\nINT8 vs FP8 per-tensor și per-channel")
    ax.set_ylabel("Δ Accuracy (pp)")
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.003 if val >= 0 else -0.008),
                f"{val:+.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "01_degradation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 02 Grouped: INT8 vs FP8 per granularity ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = ["Per-tensor", "Per-channel"]
    int8_accs = [results["INT8-pt"]["accuracy_percent"],
                 results["INT8-pc"]["accuracy_percent"]]
    fp8_accs  = [results["FP8-pt"]["accuracy_percent"],
                 results["FP8-pc"]["accuracy_percent"]]
    x = np.arange(2)
    w = 0.3
    b1 = ax.bar(x - w / 2, int8_accs, w, label="INT8",
                color=["#E69F00", "#D55E00"], edgecolor="white")
    b2 = ax.bar(x + w / 2, fp8_accs,  w, label="FP8 E4M3FN",
                color=["#CC79A7", "#009E73"], edgecolor="white")
    ax.axhline(fp32_acc, color=COLORS["FP32"], linewidth=1.5, linestyle="--",
               label=f"FP32 = {fp32_acc:.3f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("INT8 vs FP8 E4M3FN\nper granularitate (per-tensor / per-channel)")
    ax.legend()
    ax.set_ylim(74.5, 75.8)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}%",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "02_int8_vs_fp8_grouped.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 03 Tradeoff scatter ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in methods:
        ax.scatter(results[m]["memory_mb"], results[m]["accuracy_percent"],
                   color=COLORS.get(m, "#999"), s=180, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(m,
                    (results[m]["memory_mb"] + 0.15,
                     results[m]["accuracy_percent"] + 0.01),
                    fontsize=10, color=COLORS.get(m, "#999"), fontweight="bold")
    ax.axhline(fp32_acc, color=COLORS["FP32"], linewidth=1,
               linestyle=":", alpha=0.6, label="FP32 baseline")
    ax.set_xlabel("Memory (MB)  ← mai mic e mai bun")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Tradeoff: Acuratețe vs Memorie\nINT8 vs FP8 E4M3FN")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "03_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("  Plots saved:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"    {p}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    out_dir = Path("results/experiments/fp8_vs_int8/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 70)
    print("Experiment: FP8 E4M3FN vs INT8 — weight-only quantization")
    print(f"PyTorch {torch.__version__}  |  device: {device}")
    print(f"FP8 dtype: torch.float8_e4m3fn  (max={FP8_MAX})")
    print("=" * 70)

    base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    data_cfg   = timm.data.resolve_data_config({}, model=base_model)
    transform  = timm.data.create_transform(**data_cfg)

    print("\nLoading ImageNet-1k validation...")
    loader = load_imagenet_val(transform, batch_size=64)

    configs = [
        ("FP32",    lambda m: m,                                                       "FP32 baseline"),
        ("INT8-pt", lambda m: quantize_model_selective(m, verbose=False)[0],           "INT8 per-tensor (Faza 2)"),
        ("INT8-pc", lambda m: quantize_model_per_channel(m, verbose=False)[0],        "INT8 per-channel (Faza 3)"),
        ("FP8-pt",  lambda m: quantize_model_fp8(m, per_channel=False, verbose=True)[0], "FP8 E4M3FN per-tensor"),
        ("FP8-pc",  lambda m: quantize_model_fp8(m, per_channel=True,  verbose=True)[0], "FP8 E4M3FN per-channel"),
    ]

    results: dict = {}

    for label, quantize_fn, desc in configs:
        print(f"\n{'='*70}")
        print(f"{label}: {desc}")
        print("=" * 70)
        model = copy.deepcopy(base_model).to(device)
        model = quantize_fn(model)
        mem   = model_size_mb(model)
        disk  = disk_size_mb(model)
        n_q   = count_quant_layers(model)
        print(f"  Layers quantized: {n_q}  |  Memory: {mem} MB  |  Disk: {disk} MB")
        res = evaluate(model, loader, device, label=label)
        results[label] = {**res, "memory_mb": mem, "disk_mb": disk,
                          "n_layers_quant": n_q}
        print(f"  Accuracy: {res['accuracy_percent']}%  |  "
              f"Latency: {res['avg_latency_ms_per_batch']} ms/batch")
        del model

    # ── Summary ─────────────────────────────────────────────────────────
    fp32_acc = results["FP32"]["accuracy_percent"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<12} {'Accuracy':>10} {'Δ FP32':>10} "
          f"{'Mem (MB)':>10} {'Disk (MB)':>10} {'Lat (ms)':>12}")
    print("-" * 70)
    for label, _, _ in configs:
        r     = results[label]
        delta = r["accuracy_percent"] - fp32_acc
        print(f"{label:<12} {r['accuracy_percent']:>9.3f}%  "
              f"{delta:>+9.4f}  {r['memory_mb']:>9.2f}  "
              f"{r['disk_mb']:>9.2f}  "
              f"{r['avg_latency_ms_per_batch']:>11.1f}")

    # ── Verdict ──────────────────────────────────────────────────────────
    fp8pt_deg  = results["FP8-pt"]["accuracy_percent"] - fp32_acc
    fp8pc_deg  = results["FP8-pc"]["accuracy_percent"] - fp32_acc
    int8pt_deg = results["INT8-pt"]["accuracy_percent"] - fp32_acc
    int8pc_deg = results["INT8-pc"]["accuracy_percent"] - fp32_acc

    print("\nVERDICT:")
    print(f"  FP8-pt vs INT8-pt:  {fp8pt_deg - int8pt_deg:+.4f} pp "
          f"({'FP8 mai bun' if fp8pt_deg > int8pt_deg else 'INT8 mai bun'})")
    print(f"  FP8-pc vs INT8-pc:  {fp8pc_deg - int8pc_deg:+.4f} pp "
          f"({'FP8 mai bun' if fp8pc_deg > int8pc_deg else 'INT8 mai bun'})")

    output = {
        "experiment":  "fp8_vs_int8",
        "hypothesis":  "FP8 E4M3FN (with proper linear scaling) vs INT8 weight-only quantization",
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":       "vit_tiny_patch16_224",
        "dataset":     "ImageNet-1k validation (50000 images)",
        "device":      str(device),
        "fp8_dtype":   "float8_e4m3fn",
        "fp8_max":     FP8_MAX,
        "results":     results,
        "verdict": {
            "fp8pt_vs_int8pt_pp": round(fp8pt_deg - int8pt_deg, 4),
            "fp8pc_vs_int8pc_pp": round(fp8pc_deg - int8pc_deg, 4),
            "fp8_better_than_int8": fp8pt_deg > int8pt_deg or fp8pc_deg > int8pc_deg,
        },
    }

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")

    generate_plots(results)
    print("=" * 70)


if __name__ == "__main__":
    main()
