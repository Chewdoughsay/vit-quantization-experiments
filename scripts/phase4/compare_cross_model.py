"""
Phase 4 — Cross-model comparison.

Loads results from all three model sizes and generates comparison plots
showing how quantization scales with model size.

Run AFTER evaluate_model.py has been executed for all three models.

Usage:
    python scripts/phase4/compare_cross_model.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

MODELS = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
]

MODEL_LABELS = {
    "vit_tiny_patch16_224":  "ViT-Tiny\n(5.7M)",
    "vit_small_patch16_224": "ViT-Small\n(22M)",
    "vit_base_patch16_224":  "ViT-Base\n(86M)",
}

METHODS  = ["FP32", "FP16", "INT8-pt", "INT8-pc"]
COLORS   = {
    "FP32":    "#0072B2",
    "FP16":    "#009E73",
    "INT8-pt": "#E69F00",
    "INT8-pc": "#D55E00",
}

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "#F8F8F8",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.4, "grid.linestyle": "--",
})


def load_results() -> dict:
    data = {}
    for model in MODELS:
        path = Path(f"results/Phase4/{model}/metrics/results.json")
        if not path.exists():
            print(f"  WARNING: {path} not found — skipping {model}")
            continue
        with open(path) as f:
            data[model] = json.load(f)
    return data


def plot_accuracy_comparison(data: dict, out_dir: Path) -> None:
    """Grouped bar chart: accuracy per model × method."""
    models_avail = [m for m in MODELS if m in data]
    x = np.arange(len(models_avail))
    width = 0.18
    offsets = np.linspace(-0.27, 0.27, len(METHODS))

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, method in zip(offsets, METHODS):
        accs = [data[m][method]["accuracy_percent"] for m in models_avail]
        ax.bar(x + offset, accs, width, label=method, color=COLORS[method])

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models_avail])
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs Model Size — All Formats")
    ax.legend(title="Format")

    # Annotate with raw values
    for patch in ax.patches:
        h = patch.get_height()
        ax.text(patch.get_x() + patch.get_width() / 2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "00_accuracy_cross_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 00_accuracy_cross_model.png")


def plot_degradation_heatmap(data: dict, out_dir: Path) -> None:
    """Heatmap: degradation (pp) per model × method."""
    models_avail = [m for m in MODELS if m in data]
    methods_no_fp32 = [m for m in METHODS if m != "FP32"]

    matrix = []
    for model in models_avail:
        fp32_acc = data[model]["FP32"]["accuracy_percent"]
        row = [data[model][m]["accuracy_percent"] - fp32_acc
               for m in methods_no_fp32]
        matrix.append(row)

    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.5, vmax=0.1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Δ Accuracy (pp)")

    ax.set_xticks(range(len(methods_no_fp32)))
    ax.set_xticklabels(methods_no_fp32)
    ax.set_yticks(range(len(models_avail)))
    ax.set_yticklabels([MODEL_LABELS[m].replace("\n", " ") for m in models_avail])
    ax.set_title("Accuracy Degradation Heatmap (pp vs FP32)")

    for i in range(len(models_avail)):
        for j in range(len(methods_no_fp32)):
            ax.text(j, i, f"{matrix[i, j]:+.3f}",
                    ha="center", va="center", fontsize=10,
                    color="black" if abs(matrix[i, j]) < 0.3 else "white")

    plt.tight_layout()
    fig.savefig(out_dir / "01_degradation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01_degradation_heatmap.png")


def plot_memory_scaling(data: dict, out_dir: Path) -> None:
    """Line chart: memory per format as model size grows."""
    models_avail = [m for m in MODELS if m in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in METHODS:
        mems = [data[m][method]["memory_mb"] for m in models_avail]
        ax.plot([MODEL_LABELS[m].replace("\n", " ") for m in models_avail],
                mems, marker="o", color=COLORS[method], label=method, linewidth=2)

    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Footprint vs Model Size")
    ax.legend(title="Format")
    plt.tight_layout()
    fig.savefig(out_dir / "02_memory_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 02_memory_scaling.png")


def plot_latency_scaling(data: dict, out_dir: Path) -> None:
    """Line chart: latency per format as model size grows."""
    models_avail = [m for m in MODELS if m in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in METHODS:
        lats = [data[m][method]["avg_latency_ms_per_batch"] for m in models_avail]
        ax.plot([MODEL_LABELS[m].replace("\n", " ") for m in models_avail],
                lats, marker="o", color=COLORS[method], label=method, linewidth=2)

    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title("Inference Latency vs Model Size")
    ax.legend(title="Format")
    plt.tight_layout()
    fig.savefig(out_dir / "03_latency_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 03_latency_scaling.png")


def write_summary_table(data: dict, out_dir: Path) -> None:
    """Write markdown summary table."""
    models_avail = [m for m in MODELS if m in data]
    lines = [
        "# Phase 4 — Cross-Model Comparison Summary\n",
        "| Model | Format | Accuracy (%) | Δ FP32 (pp) | Memory (MB) | Latency (ms) |",
        "|-------|--------|-------------|------------|------------|-------------|",
    ]
    for model in models_avail:
        fp32_acc = data[model]["FP32"]["accuracy_percent"]
        for method in METHODS:
            r = data[model][method]
            delta = r["accuracy_percent"] - fp32_acc
            lines.append(
                f"| {MODEL_LABELS[model].replace(chr(10), ' ')} "
                f"| {method} | {r['accuracy_percent']:.3f} "
                f"| {delta:+.4f} | {r['memory_mb']:.2f} "
                f"| {r['avg_latency_ms_per_batch']:.1f} |"
            )

    out_path = out_dir / "summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


def main():
    out_dir = Path("results/Phase4/cross_model_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_results()
    if not data:
        print("No results found. Run evaluate_model.py first.")
        return

    print(f"Loaded {len(data)} model(s): {list(data.keys())}\n")
    print("Generating plots...")

    plot_accuracy_comparison(data, out_dir)
    plot_degradation_heatmap(data, out_dir)
    plot_memory_scaling(data, out_dir)
    plot_latency_scaling(data, out_dir)
    write_summary_table(data, out_dir)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
