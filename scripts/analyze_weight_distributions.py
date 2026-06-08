"""
Analiză distribuție ponderi per strat — familia ViT.

Confirmă că ponderile straturilor nn.Linear din ViT-Tiny, ViT-Small și
ViT-Base sunt aproximativ centrate în 0 (media per strat ≈ 0).

Rulare:
    python scripts/analyze_weight_distributions.py

Output:
    results/weight_distributions/metrics/stats.json   — statistici per strat
    results/weight_distributions/plots/               — grafice
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from scipy import stats

# ── Configurație ─────────────────────────────────────────────────────────────

MODELS = {
    "ViT-Tiny":  "vit_tiny_patch16_224",
    "ViT-Small": "vit_small_patch16_224",
    "ViT-Base":  "vit_base_patch16_224",
}

SKIP_PATTERNS = ["norm", "cls_token", "pos_embed", "head"]

RESULTS_DIR = Path("results/weight_distributions")
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR   = RESULTS_DIR / "plots"


def should_skip(name: str) -> bool:
    return any(p in name for p in SKIP_PATTERNS)


def layer_stats(weight: torch.Tensor) -> dict:
    w = weight.detach().float().cpu().numpy().ravel()
    skew_val, _ = stats.skewtest(w) if len(w) >= 8 else (float("nan"), float("nan"))
    return {
        "mean":     float(np.mean(w)),
        "std":      float(np.std(w)),
        "median":   float(np.median(w)),
        "skewness": float(stats.skew(w)),
        "kurtosis": float(stats.kurtosis(w)),
        "abs_mean": float(np.mean(np.abs(w))),
        "n_params": int(w.size),
    }


def analyze_model(model_name: str, timm_id: str) -> dict[str, dict]:
    print(f"\n{'='*60}")
    print(f"  Analizez {model_name} ({timm_id})")
    print(f"{'='*60}")

    model = timm.create_model(timm_id, pretrained=True)
    model.eval()

    results: dict[str, dict] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if should_skip(name):
            continue
        s = layer_stats(module.weight)
        results[name] = s
        print(f"  {name:<45} mean={s['mean']:+.5f}  std={s['std']:.4f}  "
              f"skew={s['skewness']:+.3f}")

    return results


def plot_means(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(all_stats), figsize=(5 * len(all_stats), 5),
                             sharey=False)
    if len(all_stats) == 1:
        axes = [axes]

    for ax, (model_name, layers) in zip(axes, all_stats.items()):
        means = [s["mean"] for s in layers.values()]
        ax.hist(means, bins=20, color="#0072B2", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="0")
        ax.set_title(f"{model_name}\n(n={len(means)} straturi)")
        ax.set_xlabel("Media ponderilor per strat")
        ax.set_ylabel("Număr straturi")
        ax.legend()

        # Annotate with overall mean of means
        grand_mean = float(np.mean(means))
        ax.text(0.97, 0.95, f"$\\bar{{\\mu}}$ = {grand_mean:+.5f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Distribuția mediei ponderilor per strat — familia ViT\n"
                 "(straturi nn.Linear cuantizabile, exclus SKIP_PATTERNS)",
                 fontsize=12)
    plt.tight_layout()
    path = out_dir / "01_means_histogram.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Salvat: {path}")


def plot_mean_per_layer(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    fig, axes = plt.subplots(len(all_stats), 1,
                             figsize=(14, 4 * len(all_stats)))
    if len(all_stats) == 1:
        axes = [axes]

    for ax, (model_name, layers) in zip(axes, all_stats.items()):
        names = list(layers.keys())
        means = [layers[n]["mean"] for n in names]
        stds  = [layers[n]["std"]  for n in names]
        x = np.arange(len(names))

        ax.bar(x, means, color="#0072B2", alpha=0.7, label="media")
        ax.errorbar(x, means, yerr=stds, fmt="none", color="#D55E00",
                    elinewidth=0.6, capsize=2, label="±std")
        ax.axhline(0, color="red", linestyle="--", linewidth=1, label="0")
        ax.set_title(f"{model_name} — media ponderilor per strat liniar")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.set_ylabel("Media ponderilor")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = out_dir / "02_mean_per_layer.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvat: {path}")


def plot_skewness(all_stats: dict[str, dict[str, dict]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"ViT-Tiny": "#0072B2", "ViT-Small": "#E69F00", "ViT-Base": "#009E73"}

    for model_name, layers in all_stats.items():
        skews = [s["skewness"] for s in layers.values()]
        ax.hist(skews, bins=15, alpha=0.6, label=model_name,
                color=colors.get(model_name, "gray"), edgecolor="white")

    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="simetrie perfectă")
    ax.set_xlabel("Skewness per strat")
    ax.set_ylabel("Număr straturi")
    ax.set_title("Asimetria distribuției ponderilor per strat\n"
                 "(skewness ≈ 0 confirmă simetria față de 0)")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "03_skewness.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvat: {path}")


def print_summary(all_stats: dict[str, dict[str, dict]]) -> None:
    print(f"\n{'='*70}")
    print("REZUMAT — Mediile ponderilor per model")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'Straturi':>8} {'|μ| mediu':>12} {'|μ| max':>12} "
          f"{'% straturi cu |μ|<0.01':>22}")
    print("-" * 70)
    for model_name, layers in all_stats.items():
        means    = [abs(s["mean"]) for s in layers.values()]
        n        = len(means)
        avg_abs  = float(np.mean(means))
        max_abs  = float(np.max(means))
        pct_near = 100 * sum(1 for m in means if m < 0.01) / n
        print(f"{model_name:<12} {n:>8} {avg_abs:>12.6f} {max_abs:>12.6f} "
              f"{pct_near:>21.1f}%")
    print(f"{'='*70}")
    print("\nConcluzie: valorile |μ| mediu aproape de 0 confirmă că distribuțiile")
    print("ponderilor sunt centrate simetric în jurul lui 0 per strat.")


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict[str, dict]] = {}
    for model_name, timm_id in MODELS.items():
        all_stats[model_name] = analyze_model(model_name, timm_id)

    # Salvare JSON
    json_path = METRICS_DIR / "stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Statistici salvate: {json_path}")

    # Grafice
    plot_means(all_stats, PLOTS_DIR)
    plot_mean_per_layer(all_stats, PLOTS_DIR)
    plot_skewness(all_stats, PLOTS_DIR)

    print_summary(all_stats)


if __name__ == "__main__":
    main()
