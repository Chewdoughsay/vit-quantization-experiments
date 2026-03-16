"""
Generare ploturi raport Faza 1 — FP16 static pe ImageNette.

Citește results/FP16ImageNet/metrics/fp16_imagenet_results.json
și salvează toate figurile în results/FP16ImageNet/plots/.

Rulare: python scripts/generate_fp16_imagenet_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configurare globală stil (identic cu generate_report_plots.py)
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "DejaVu Serif",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    13,
    "legend.fontsize":   11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F8F8",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.4,
    "grid.linestyle":    "--",
})

SAVE_DPI = 300

COLORS = {
    "FP32": "#0072B2",   # albastru
    "FP16": "#009E73",   # teal
}

RESULTS_PATH = Path("results/FP16ImageNet/metrics/fp16_imagenet_results.json")
OUTPUT_DIR   = Path("results/FP16ImageNet/plots")


def load_results() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def save(fig: plt.Figure, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Plot 1: Accuracy comparison  (bar chart)
# ---------------------------------------------------------------------------

def plot_accuracy(data: dict) -> None:
    fp32_acc = data["fp32"]["accuracy_percent"]
    fp16_acc = data["fp16"]["accuracy_percent"]

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ["FP32 (baseline)", "FP16 (model.half())"],
        [fp32_acc, fp16_acc],
        color=[COLORS["FP32"], COLORS["FP16"]],
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )

    # Valori deasupra barelor
    for bar, val in zip(bars, [fp32_acc, fp16_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center", va="bottom",
            fontweight="bold", fontsize=12,
        )

    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy: FP32 vs FP16\nViT-Tiny pretrained on ImageNet-1k (ImageNette)")
    ax.set_ylim(70, 85)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")

    degradation = data["comparison"]["accuracy_degradation_pp"]
    sign = "+" if degradation > 0 else ""
    ax.text(
        0.98, 0.04,
        f"Δ = {sign}{degradation:.4f} pp",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=11, color="gray",
    )

    save(fig, "01_accuracy_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2: Memory footprint  (bar chart)
# ---------------------------------------------------------------------------

def plot_memory(data: dict) -> None:
    fp32_mem = data["fp32"]["memory_mb"]
    fp16_mem = data["fp16"]["memory_mb"]

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ["FP32 (float32)", "FP16 (float16)"],
        [fp32_mem, fp16_mem],
        color=[COLORS["FP32"], COLORS["FP16"]],
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )

    for bar, val in zip(bars, [fp32_mem, fp16_mem]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f} MB",
            ha="center", va="bottom",
            fontweight="bold", fontsize=12,
        )

    ax.set_ylabel("Model Parameter Memory (MB)")
    ax.set_title("Memory Footprint: FP32 vs FP16\nViT-Tiny (5.7M parameters)")
    ax.set_ylim(0, 28)

    ratio = data["comparison"]["memory_reduction_ratio"]
    ax.text(
        0.98, 0.96,
        f"Reduction: {ratio:.2f}×\n(4 bytes → 2 bytes / param)",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11, color="gray",
    )

    save(fig, "02_memory_comparison.png")


# ---------------------------------------------------------------------------
# Plot 3: Inference latency  (bar chart)
# ---------------------------------------------------------------------------

def plot_latency(data: dict) -> None:
    lat_fp32 = data["fp32"]["avg_latency_ms_per_batch"]
    lat_fp16 = data["fp16"]["avg_latency_ms_per_batch"]
    speedup  = data["comparison"]["latency_speedup"]
    bs       = data["batch_size"]

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ["FP32 (float32)", "FP16 (float16)"],
        [lat_fp32, lat_fp16],
        color=[COLORS["FP32"], COLORS["FP16"]],
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )

    for bar, val in zip(bars, [lat_fp32, lat_fp16]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f} ms",
            ha="center", va="bottom",
            fontweight="bold", fontsize=12,
        )

    ax.set_ylabel(f"Avg. Latency per Batch (ms)  [bs={bs}]")
    ax.set_title("Inference Latency: FP32 vs FP16\nApple M4 MPS backend")
    ax.set_ylim(0, 180)

    ax.text(
        0.98, 0.96,
        f"Speedup: {speedup:.2f}×",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=12, fontweight="bold", color=COLORS["FP16"],
    )

    save(fig, "03_latency_comparison.png")


# ---------------------------------------------------------------------------
# Plot 4: Summary overview  (3 subplots pe un singur canvas)
# ---------------------------------------------------------------------------

def plot_summary(data: dict) -> None:
    fp32_acc = data["fp32"]["accuracy_percent"]
    fp16_acc = data["fp16"]["accuracy_percent"]
    fp32_mem = data["fp32"]["memory_mb"]
    fp16_mem = data["fp16"]["memory_mb"]
    lat_fp32 = data["fp32"]["avg_latency_ms_per_batch"]
    lat_fp16 = data["fp16"]["avg_latency_ms_per_batch"]
    speedup  = data["comparison"]["latency_speedup"]
    deg      = data["comparison"]["accuracy_degradation_pp"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "FP16 Static Quantization — ViT-Tiny on ImageNette\n"
        "(pretrained ImageNet-1k, Apple M4 MPS)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    labels = ["FP32", "FP16"]
    colors = [COLORS["FP32"], COLORS["FP16"]]

    # -- Accuracy
    ax = axes[0]
    bars = ax.bar(labels, [fp32_acc, fp16_acc], color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [fp32_acc, fp16_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Top-1 Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 86)
    sign = "+" if deg > 0 else ""
    ax.text(0.97, 0.04, f"Δ = {sign}{deg:.4f} pp",
            transform=ax.transAxes, ha="right", va="bottom", color="gray", fontsize=10)

    # -- Memory
    ax = axes[1]
    bars = ax.bar(labels, [fp32_mem, fp16_mem], color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [fp32_mem, fp16_mem]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f} MB", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Parameter Memory")
    ax.set_ylabel("Memory (MB)")
    ax.set_ylim(0, 28)
    ax.text(0.97, 0.96, "2.00× reduction",
            transform=ax.transAxes, ha="right", va="top", color="gray", fontsize=10)

    # -- Latency
    ax = axes[2]
    bars = ax.bar(labels, [lat_fp32, lat_fp16], color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [lat_fp32, lat_fp16]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f} ms", ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Inference Latency (bs={data['batch_size']})")
    ax.set_ylabel("Avg. Latency per Batch (ms)")
    ax.set_ylim(0, 185)
    ax.text(0.97, 0.96, f"{speedup:.2f}× speedup",
            transform=ax.transAxes, ha="right", va="top",
            color=COLORS["FP16"], fontsize=10, fontweight="bold")

    fig.tight_layout()
    save(fig, "00_summary.png")


# ---------------------------------------------------------------------------
# Plot 5: Accuracy-Latency tradeoff scatter
# ---------------------------------------------------------------------------

def plot_tradeoff(data: dict) -> None:
    points = {
        "FP32": (data["fp32"]["avg_latency_ms_per_batch"],
                 data["fp32"]["accuracy_percent"]),
        "FP16": (data["fp16"]["avg_latency_ms_per_batch"],
                 data["fp16"]["accuracy_percent"]),
    }

    fig, ax = plt.subplots(figsize=(6, 5))

    for label, (lat, acc) in points.items():
        ax.scatter(lat, acc, s=180, color=COLORS[label],
                   zorder=5, label=label, edgecolors="white", linewidths=1.5)
        ax.annotate(
            f"  {label}\n  {acc:.2f}% | {lat:.0f} ms",
            xy=(lat, acc),
            fontsize=10,
            color=COLORS[label],
        )

    ax.set_xlabel("Avg. Inference Latency per Batch (ms)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy–Latency Tradeoff\nFP32 vs FP16 (model.half())")
    ax.legend()

    save(fig, "04_accuracy_latency_tradeoff.png")


# ---------------------------------------------------------------------------
# Text summary (Markdown)
# ---------------------------------------------------------------------------

def write_markdown(data: dict) -> None:
    deg      = data["comparison"]["accuracy_degradation_pp"]
    speedup  = data["comparison"]["latency_speedup"]
    mem_ratio = data["comparison"]["memory_reduction_ratio"]

    lines = [
        "# Faza 1 — FP16 Static Quantization pe ImageNette",
        "",
        f"**Data:** {data['timestamp']}  |  "
        f"**Model:** `{data['model']}`  |  "
        f"**Dataset:** {data['dataset']}",
        f"**Device:** {data['device'].upper()} (Apple M4)  |  "
        f"**PyTorch:** {data['pytorch_version']}",
        "",
        "## Rezultate",
        "",
        "| Metrică | FP32 | FP16 | Δ |",
        "|---------|------|------|---|",
        f"| Top-1 Accuracy | {data['fp32']['accuracy_percent']:.4f}% | "
        f"{data['fp16']['accuracy_percent']:.4f}% | "
        f"{'%+.4f pp' % (-deg)} |",
        f"| Loss | {data['fp32']['avg_loss']:.4f} | "
        f"{data['fp16']['avg_loss']:.4f} | — |",
        f"| Latency/batch | {data['fp32']['avg_latency_ms_per_batch']:.1f} ms | "
        f"{data['fp16']['avg_latency_ms_per_batch']:.1f} ms | "
        f"**{speedup:.2f}×** mai rapid |",
        f"| Memorie model | {data['fp32']['memory_mb']:.2f} MB | "
        f"{data['fp16']['memory_mb']:.2f} MB | "
        f"**{mem_ratio:.2f}×** mai mic |",
        "",
        "## Concluzii",
        "",
        f"- Degradarea de acuratețe FP32→FP16: **{deg:+.4f} pp** (neglijabilă)",
        f"- Reducerea de memorie: **{mem_ratio:.2f}×** (exact conform teoriei: 4 bytes → 2 bytes)",
        f"- Speedup inferență pe MPS: **{speedup:.2f}×** (Apple M4 valorifică float16 nativ)",
        "- `model.half()` funcționează complet pe MPS (PyTorch 2.9), fără fallback CPU",
        "- timm `vit_tiny_patch16_224` folosește normalizare **(0.5, 0.5, 0.5)**, "
        "nu ImageNet standard — important de reținut pentru experimente ulterioare",
        "",
        "## Fișiere generate",
        "",
        "| Fișier | Conținut |",
        "|--------|----------|",
        "| `00_summary.png` | Overview 3 metrici pe un canvas |",
        "| `01_accuracy_comparison.png` | Acuratețe FP32 vs FP16 |",
        "| `02_memory_comparison.png` | Memorie FP32 vs FP16 |",
        "| `03_latency_comparison.png` | Latență inferență |",
        "| `04_accuracy_latency_tradeoff.png` | Scatter accuracy-latency |",
    ]

    path = Path("results/FP16ImageNet") / "summary.md"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nGenerare ploturi Faza 1 — FP16 ImageNet\n")
    data = load_results()

    plot_summary(data)
    plot_accuracy(data)
    plot_memory(data)
    plot_latency(data)
    plot_tradeoff(data)
    write_markdown(data)

    print(f"\nToate figurile salvate în: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
