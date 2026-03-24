"""
Phase 2 report plots — INT8 static quantization on ImageNet-1k.

Reads:
    results/INT8ImageNet/metrics/int8_imagenet_results.json
    results/FP16ImageNet/metrics/fp16_imagenet_results.json  (for cross-phase comparison)

Outputs:
    results/INT8ImageNet/plots/

Usage:
    python scripts/generate_int8_imagenet_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.plot_style import apply_style, COLORS, save_fig

apply_style()

INT8_RESULTS_PATH = Path("results/INT8ImageNet/metrics/int8_imagenet_results.json")
FP16_RESULTS_PATH = Path("results/FP16ImageNet/metrics/fp16_imagenet_results.json")
OUTPUT_DIR        = Path("results/INT8ImageNet/plots")


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1: Summary 3-panel  (FP32 vs INT8)
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary(d: dict) -> None:
    fp32 = d["fp32"]
    int8 = d["int8"]
    comp = d["comparison"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "INT8 Static Quantization (per-tensor) — ViT-Tiny on ImageNet-1k\n"
        "(weight-only, linear scaling, Apple M4 MPS)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    labels = ["FP32", "INT8"]
    colors = [COLORS["FP32"], COLORS["INT8"]]

    # Accuracy
    ax = axes[0]
    bars = ax.bar(labels, [fp32["accuracy_percent"], int8["accuracy_percent"]],
                  color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [fp32["accuracy_percent"], int8["accuracy_percent"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Top-1 Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 86)
    deg = comp["accuracy_degradation_pp"]
    ax.text(0.97, 0.04, f"Δ = +{deg:.4f} pp",
            transform=ax.transAxes, ha="right", va="bottom", color="gray", fontsize=10)

    # Memory
    ax = axes[1]
    bars = ax.bar(labels, [fp32["memory_mb"], int8["memory_mb"]],
                  color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, [fp32["memory_mb"], int8["memory_mb"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f} MB", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Parameter + Buffer Memory")
    ax.set_ylabel("Memory (MB)")
    ax.set_ylim(0, 28)
    ax.text(0.97, 0.96, f"{comp['memory_reduction_ratio']:.2f}× reduction",
            transform=ax.transAxes, ha="right", va="top", color="gray", fontsize=10)

    # Latency
    ax = axes[2]
    lat_vals = [fp32["avg_latency_ms_per_batch"], int8["avg_latency_ms_per_batch"]]
    bars = ax.bar(labels, lat_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, lat_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f} ms", ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Inference Latency (bs={d['batch_size']})")
    ax.set_ylabel("Avg. Latency per Batch (ms)")
    ax.set_ylim(0, 185)
    spd = comp["latency_speedup"]
    note = f"{spd:.2f}× speedup" if spd and spd > 1 else "no speedup\n(dequant overhead)"
    ax.text(0.97, 0.96, note, transform=ax.transAxes, ha="right", va="top",
            color=COLORS["INT8"], fontsize=10, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "00_summary.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2: Per-layer MSE bar chart
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_layer_mse(d: dict) -> None:
    stats  = d["layer_stats"]
    names  = [s["layer"] for s in stats]
    mse    = [s["mse"]   for s in stats]

    short_names = []
    for n in names:
        n = n.replace("blocks.", "b").replace(".attn.", ".attn.").replace(".mlp.", ".mlp.")
        short_names.append(n)

    colors_bars = [
        "#D55E00" if v > np.percentile(mse, 90) else COLORS["INT8"]
        for v in mse
    ]

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(names))
    ax.bar(x, mse, color=colors_bars, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("MSE (weight quantization error)")
    ax.set_title("Per-Layer Weight Quantization Error (MSE) — INT8 per-tensor\n"
                 "Orange = outliers (>P90)")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    avg_mse = np.mean(mse)
    ax.axhline(avg_mse, color="gray", linestyle="--", linewidth=1.2,
               label=f"Mean = {avg_mse:.2e}")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "01_per_layer_mse.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3: Per-layer scale values (distribution)
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_layer_scale(d: dict) -> None:
    stats  = d["layer_stats"]
    scales = [s["scale"] for s in stats]

    groups = {"attn.qkv": [], "attn.proj": [], "mlp.fc1": [], "mlp.fc2": []}
    for s in stats:
        for key in groups:
            if key in s["layer"]:
                groups[key].append(s["scale"])
                break

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scale Factor Distribution per Layer — INT8 per-tensor",
                 fontsize=14, fontweight="bold")

    # Left: scatter all layers
    ax = axes[0]
    ax.scatter(range(len(scales)), scales, color=COLORS["INT8"], s=50, zorder=5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Scale factor (max(|W|) / 127)")
    ax.set_title("Scale per layer (model order)")
    ax.axhline(np.mean(scales), color="gray", linestyle="--", linewidth=1,
               label=f"Mean = {np.mean(scales):.5f}")
    ax.legend()

    # Right: boxplot per layer type
    ax = axes[1]
    group_colors = ["#0072B2", "#56B4E9", "#D55E00", "#009E73"]
    bp = ax.boxplot(
        [groups[k] for k in groups],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(list(groups.keys()), rotation=20, ha="right")
    ax.set_ylabel("Scale factor")
    ax.set_title("Scale per layer type")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "02_per_layer_scale.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4: Cross-phase comparison  FP32 / FP16 / INT8
# ═══════════════════════════════════════════════════════════════════════════

def plot_cross_phase(d_int8: dict, d_fp16: dict) -> None:
    methods = ["FP32", "FP16", "INT8"]
    colors  = [COLORS["FP32"], COLORS["FP16"], COLORS["INT8"]]

    acc_vals = [
        d_int8["fp32"]["accuracy_percent"],
        d_fp16["fp16"]["accuracy_percent"],
        d_int8["int8"]["accuracy_percent"],
    ]
    mem_vals = [
        d_int8["fp32"]["memory_mb"],
        d_fp16["fp16"]["memory_mb"],
        d_int8["int8"]["memory_mb"],
    ]
    lat_vals = [
        d_int8["fp32"]["avg_latency_ms_per_batch"],
        d_fp16["fp16"]["avg_latency_ms_per_batch"],
        d_int8["int8"]["avg_latency_ms_per_batch"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Comparison FP32 / FP16 / INT8 — ViT-Tiny on ImageNet-1k\n"
        "(FP16 = model.half()  |  INT8 = per-tensor weight-only)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Accuracy
    ax = axes[0]
    bars = ax.bar(methods, acc_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Top-1 Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(70, 86)

    # Memory
    ax = axes[1]
    bars = ax.bar(methods, mem_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, mem_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f} MB", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Parameter Memory")
    ax.set_ylabel("Memory (MB)")
    ax.set_ylim(0, 28)

    # Latency
    ax = axes[2]
    bars = ax.bar(methods, lat_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, lat_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f} ms", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title(f"Inference Latency (bs={d_int8['batch_size']})")
    ax.set_ylabel("Avg. Latency per Batch (ms)")
    ax.set_ylim(0, 185)

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "03_cross_phase_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 5: Accuracy–Memory tradeoff scatter
# ═══════════════════════════════════════════════════════════════════════════

def plot_tradeoff(d_int8: dict, d_fp16: dict) -> None:
    points = {
        "FP32": (d_int8["fp32"]["memory_mb"],  d_int8["fp32"]["accuracy_percent"]),
        "FP16": (d_fp16["fp16"]["memory_mb"],  d_fp16["fp16"]["accuracy_percent"]),
        "INT8": (d_int8["int8"]["memory_mb"],  d_int8["int8"]["accuracy_percent"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Accuracy–Efficiency Tradeoff — FP32 / FP16 / INT8",
                 fontsize=14, fontweight="bold")

    # Left: Accuracy vs Memory
    ax = axes[0]
    for label, (mem, acc) in points.items():
        ax.scatter(mem, acc, s=220, color=COLORS[label], zorder=5,
                   label=label, edgecolors="white", linewidths=1.8)
        ax.annotate(f"  {label}\n  {acc:.2f}% | {mem:.1f} MB",
                    xy=(mem, acc), fontsize=10, color=COLORS[label])
    ax.set_xlabel("Parameter Memory (MB)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs Memory")
    ax.legend()

    # Right: Accuracy vs Latency
    lat_points = {
        "FP32": (d_int8["fp32"]["avg_latency_ms_per_batch"], d_int8["fp32"]["accuracy_percent"]),
        "FP16": (d_fp16["fp16"]["avg_latency_ms_per_batch"], d_fp16["fp16"]["accuracy_percent"]),
        "INT8": (d_int8["int8"]["avg_latency_ms_per_batch"], d_int8["int8"]["accuracy_percent"]),
    }
    ax = axes[1]
    for label, (lat, acc) in lat_points.items():
        ax.scatter(lat, acc, s=220, color=COLORS[label], zorder=5,
                   label=label, edgecolors="white", linewidths=1.8)
        ax.annotate(f"  {label}\n  {acc:.2f}% | {lat:.0f} ms",
                    xy=(lat, acc), fontsize=10, color=COLORS[label])
    ax.set_xlabel("Avg. Inference Latency per Batch (ms)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs Latency")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "04_tradeoff.png")


# ═══════════════════════════════════════════════════════════════════════════
# Markdown summary
# ═══════════════════════════════════════════════════════════════════════════

def write_markdown(d: dict) -> None:
    comp  = d["comparison"]
    q     = d["quantization"]
    fp32  = d["fp32"]
    int8  = d["int8"]
    spd   = comp["latency_speedup"]

    lines = [
        "# Faza 2 — INT8 Static Quantization on ImageNet-1k",
        "",
        f"**Data:** {d['timestamp']}  |  "
        f"**Model:** `{d['model']}`  |  "
        f"**Dataset:** {d['dataset']}",
        f"**Device:** {d['device'].upper()} (Apple M4)  |  "
        f"**PyTorch:** {d['pytorch_version']}",
        "",
        "## Metodă",
        "",
        "Cuantizare **weight-only per-tensor** cu scalare liniară:",
        "```",
        "scale = max(|W|) / 127",
        "q     = round(W / scale).clamp(-128, 127).to(int8)",
        "```",
        f"- Straturi cuantizate: **{q['n_layers_quant']}** `nn.Linear` "
        f"({q['total_q_params']:,} parametri)",
        f"- Straturi excluse: `{d['quantization'].get('skip_patterns', ['norm','cls_token','pos_embed','head'])}`",
        "",
        "## Rezultate",
        "",
        "| Metrică | FP32 | INT8 | Δ |",
        "|---------|------|------|---|",
        f"| Top-1 Accuracy | {fp32['accuracy_percent']:.4f}% | "
        f"{int8['accuracy_percent']:.4f}% | "
        f"+{comp['accuracy_degradation_pp']:.4f} pp |",
        f"| Loss | {fp32['avg_loss']:.4f} | {int8['avg_loss']:.4f} | — |",
        f"| Latency/batch | {fp32['avg_latency_ms_per_batch']:.1f} ms | "
        f"{int8['avg_latency_ms_per_batch']:.1f} ms | "
        f"{'%.2f×' % spd if spd else '—'} |",
        f"| Memorie | {fp32['memory_mb']:.2f} MB | {int8['memory_mb']:.2f} MB | "
        f"**{comp['memory_reduction_ratio']:.2f}×** mai mic |",
        "",
        "## Statistici cuantizare per layer",
        "",
        f"- MSE mediu per layer: `{q['avg_mse']:.4e}`",
        f"- Worst layer: `{q['worst_layer']}` (MSE = `{q['worst_layer_mse']:.4e}`)",
        f"- Best layer:  `{q['best_layer']}` (MSE = `{q['best_layer_mse']:.4e}`)",
        "",
        "## Concluzii",
        "",
        f"- Degradarea de acuratețe: **+{comp['accuracy_degradation_pp']:.4f} pp** "
        f"(acceptabilă pentru weight-only INT8)",
        f"- Reducere memorie: **{comp['memory_reduction_ratio']:.2f}×** "
        f"(int8 = 1 byte vs float32 = 4 bytes; biasul rămâne FP32)",
        "- Fără speedup real față de FP32: INT8 dequantizează la float32 la fiecare forward "
        "→ overhead de dequantizare anulează câștigul de bandwidth",
        "- `blocks.7.mlp.*` au MSE de ~40× mai mare decât media → "
        "outlieri de ponderi, candidați pentru per-channel quantization în Faza 3",
        "",
        "## Fișiere generate",
        "",
        "| Fișier | Conținut |",
        "|--------|----------|",
        "| `00_summary.png` | Overview FP32 vs INT8 |",
        "| `01_per_layer_mse.png` | MSE per layer cuantizat |",
        "| `02_per_layer_scale.png` | Distribuție scale factors |",
        "| `03_cross_phase_comparison.png` | FP32 / FP16 / INT8 comparație |",
        "| `04_tradeoff.png` | Accuracy–Memory și Accuracy–Latency scatter |",
    ]

    path = Path("results/INT8ImageNet/summary.md")
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\nGenerare ploturi Faza 2 — INT8 ImageNet\n")

    d_int8 = load(INT8_RESULTS_PATH)
    d_fp16 = load(FP16_RESULTS_PATH)

    plot_summary(d_int8)
    plot_per_layer_mse(d_int8)
    plot_per_layer_scale(d_int8)
    plot_cross_phase(d_int8, d_fp16)
    plot_tradeoff(d_int8, d_fp16)
    write_markdown(d_int8)

    print(f"\nToate figurile salvate în: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
