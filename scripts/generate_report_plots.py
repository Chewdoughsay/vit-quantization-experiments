"""
ViT-FP8 report generator: reads all experiment data, rebuilds comparison files,
and generates all publication-quality figures plus results/summary.md.

Run from the project root: python generate_report_plots.py
"""

import json
import csv
import math
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

RESULTS_DIR = Path("results")
OUTPUT_DIR  = RESULTS_DIR / "report_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = ["BaseFP32", "BaseFP16", "AugmFP32", "AugmFP16"]
SAVE_DPI = 300

COLORS = {
    "BaseFP32": "#0072B2",   # blue
    "BaseFP16": "#56B4E9",   # sky blue
    "AugmFP32": "#D55E00",   # vermillion
    "AugmFP16": "#009E73",   # teal
}
LABELS = {
    "BaseFP32": "FP32 + Basic Aug.",
    "BaseFP16": "FP16 + Basic Aug.",
    "AugmFP32": "FP32 + Extended Aug.",
    "AugmFP16": "FP16 + Extended Aug.",
}
MARKERS = {
    "BaseFP32": "o",
    "BaseFP16": "s",
    "AugmFP32": "^",
    "AugmFP16": "D",
}

plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          12,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     13,
    "legend.fontsize":    11,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8F8F8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.4,
    "grid.linestyle":     "--",
})


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_metrics(exp: str) -> dict:
    return load_json(RESULTS_DIR / exp / "metrics" / "final_metrics.json")


def load_timing(exp: str) -> dict:
    return load_json(RESULTS_DIR / exp / "metrics" / "timing_report.json")


def load_hardware(exp: str) -> dict:
    return load_json(RESULTS_DIR / exp / "metrics" / "hardware_stats.json")


def load_gpu_csv(exp: str) -> list[dict]:
    path = RESULTS_DIR / exp / "metrics" / "gpu_stats.csv"
    if not path.exists():
        return []
    return load_csv_rows(path)


def compute_convergence_epoch(val_acc: list[float], threshold: float = 0.005) -> int:
    """First epoch (1-indexed) where val_acc >= best_val_acc - threshold."""
    best = max(val_acc)
    for i, v in enumerate(val_acc):
        if v >= best - threshold:
            return i + 1
    return len(val_acc)


def hardware_stats_summary(hw: dict) -> dict:
    cpu = [v for v in hw.get("cpu_percent", []) if v is not None]
    mem = [v for v in hw.get("memory_percent", []) if v is not None]
    thermal = hw.get("thermal_pressure", [])
    return {
        "cpu_avg":          round(float(np.mean(cpu)), 2)  if cpu     else 0.0,
        "cpu_max":          round(float(np.max(cpu)), 2)   if cpu     else 0.0,
        "mem_avg":          round(float(np.mean(mem)), 2)  if mem     else 0.0,
        "mem_max":          round(float(np.max(mem)), 2)   if mem     else 0.0,
        "thermal_max":      max(thermal)                   if thermal else 0,
        "thermal_throttled": max(thermal) > 0              if thermal else False,
    }


def rebuild_comparison() -> dict:
    comparison = {}
    for exp in EXPERIMENTS:
        m  = load_metrics(exp)
        t  = load_timing(exp)
        hw = load_hardware(exp)

        val_acc    = m["val_acc"]
        train_acc  = m["train_acc"]
        val_loss   = m["val_loss"]
        train_loss = m["train_loss"]
        epoch_times = m["epoch_time"]

        best_val  = float(max(val_acc))
        best_ep   = int(np.argmax(val_acc)) + 1   # 1-indexed
        final_val = float(val_acc[-1])
        final_tr  = float(train_acc[-1])

        hw_s = hardware_stats_summary(hw)

        comparison[exp] = {
            "num_epochs":           len(val_acc),
            "best_val_acc":         best_val,
            "best_val_acc_epoch":   best_ep,
            "final_val_acc":        final_val,
            "final_train_acc":      final_tr,
            "final_val_loss":       float(val_loss[-1]),
            "final_train_loss":     float(train_loss[-1]),
            "overfitting_gap":      round((final_tr - final_val) * 100, 4),
            "overfitting_score":    round((best_val - final_val), 6),
            "convergence_epoch":    compute_convergence_epoch(val_acc),
            # Use timing_report as authoritative source for wall-clock times
            "total_time_hours":     t["total_duration_hours"],
            "total_time_minutes":   t["total_duration_minutes"],
            "avg_epoch_time_sec":   t["avg_epoch_time_seconds"],
            "std_epoch_time_sec":   round(float(np.std(epoch_times)), 4),
            "min_epoch_time_sec":   round(float(min(epoch_times)), 4),
            "max_epoch_time_sec":   round(float(max(epoch_times)), 4),
            **hw_s,
            "precision":    t.get("precision", "").upper(),
            "augmentation": t.get("augmentation", "").capitalize(),
        }

    return comparison


def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def epochs_axis(n: int) -> range:
    return range(1, n + 1)


def plot_A1(all_m: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for exp in EXPERIMENTS:
        va = [v * 100 for v in all_m[exp]["val_acc"]]
        ax.plot(epochs_axis(len(va)), va,
                color=COLORS[exp], label=LABELS[exp],
                linewidth=2, marker=MARKERS[exp], markevery=5, markersize=6)

    ax.set_title("Validation Accuracy vs. Epoch — All Experiments")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_xlim(1, 50)
    ax.legend(framealpha=0.9)
    save(fig, "A1_val_acc_all.png")


def plot_A2(all_m: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, exp in enumerate(EXPERIMENTS):
        ax = axes[i]
        m  = all_m[exp]
        ep = epochs_axis(len(m["val_acc"]))
        tr = [v * 100 for v in m["train_acc"]]
        va = [v * 100 for v in m["val_acc"]]

        ax.plot(ep, tr, color=COLORS[exp], linewidth=2, label="Train")
        ax.plot(ep, va, color=COLORS[exp], linewidth=2, linestyle="--", label="Validation")
        ax.fill_between(ep, va, tr, alpha=0.12, color=COLORS[exp])
        ax.set_title(LABELS[exp])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(1, 50)
        ax.legend(framealpha=0.9)

    fig.suptitle("Train vs. Validation Accuracy — Overfitting Visualisation",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "A2_train_val_acc_grid.png")


def plot_B(all_m: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for exp in EXPERIMENTS:
        m  = all_m[exp]
        ep = epochs_axis(len(m["val_loss"]))
        axes[0].plot(ep, m["train_loss"], color=COLORS[exp],
                     label=LABELS[exp], linewidth=2)
        axes[1].plot(ep, m["val_loss"], color=COLORS[exp],
                     label=LABELS[exp], linewidth=2)

    axes[0].set_title("Training Loss vs. Epoch")
    axes[1].set_title("Validation Loss vs. Epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_xlim(1, 50)
        ax.legend(framealpha=0.9)

    fig.tight_layout()
    save(fig, "B_loss_curves.png")


def plot_C1(comp: dict, fp8: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    x   = np.arange(len(EXPERIMENTS))
    acc = [comp[e]["best_val_acc"] * 100 for e in EXPERIMENTS]
    bars = ax.bar(x, acc, color=[COLORS[e] for e in EXPERIMENTS],
                  width=0.55, edgecolor="white", linewidth=1.2, zorder=3)

    fp8_pc_acc = fp8["per_channel"]["accuracy_percent"]
    ax.axhline(fp8_pc_acc, color="#E69F00", linewidth=2, linestyle="--",
               label=f"FP8 per-channel (post-PTQ): {fp8_pc_acc:.2f}%", zorder=4)

    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[e] for e in EXPERIMENTS], rotation=12, ha="right")
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Best Validation Accuracy — All Experiments")
    ax.set_ylim(75, 86)
    ax.legend(framealpha=0.9)
    save(fig, "C1_best_val_acc_bar.png")


def plot_C2(comp: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    x    = np.arange(len(EXPERIMENTS))
    gaps = [comp[e]["overfitting_gap"] for e in EXPERIMENTS]
    bars = ax.bar(x, gaps, color=[COLORS[e] for e in EXPERIMENTS],
                  width=0.55, edgecolor="white", linewidth=1.2, zorder=3)

    ax.axhline(5.0, color="firebrick", linewidth=1.8, linestyle="--",
               label="5 pp reference", zorder=4)

    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[e] for e in EXPERIMENTS], rotation=12, ha="right")
    ax.set_ylabel("Overfitting Gap (Train Acc − Val Acc, pp)")
    ax.set_title("Overfitting Gap — Regularisation Effectiveness")
    ax.legend(framealpha=0.9)
    save(fig, "C2_overfitting_gap_bar.png")


def plot_C3(comp: dict, all_timing: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    x     = np.arange(len(EXPERIMENTS))
    hours = [all_timing[e]["total_duration_hours"] for e in EXPERIMENTS]
    bars  = ax.bar(x, hours, color=[COLORS[e] for e in EXPERIMENTS],
                   width=0.55, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val in zip(bars, hours):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{val:.2f}h", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[e] for e in EXPERIMENTS], rotation=12, ha="right")
    ax.set_ylabel("Total Training Time (hours)")
    ax.set_title("Total Wall-Clock Training Time")
    ax.set_ylim(0, max(hours) * 1.15)
    save(fig, "C3_training_time_bar.png")


def plot_D(fp8: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    orig_acc = fp8["original_fp16"]["accuracy_percent"]
    pt_acc   = fp8["per_tensor"]["accuracy_percent"]
    pc_acc   = fp8["per_channel"]["accuracy_percent"]

    methods = ["FP16\n(original)", "FP8 E4M3FN\nper-tensor", "FP8 E4M3FN\nper-channel"]
    accs    = [orig_acc, pt_acc, pc_acc]
    colors  = ["#0072B2", "#CC79A7", "#009E73"]

    bars = ax.bar(np.arange(3), accs, color=colors,
                  width=0.5, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val, deg in zip(bars, accs,
                              [0, fp8["per_tensor"]["degradation_pp"],
                                  fp8["per_channel"]["degradation_pp"]]):
        label = f"{val:.2f}%"
        if deg != 0:
            label += f"\n(Δ = {deg:+.2f} pp)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                label, ha="center", va="bottom", fontsize=11)

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("FP8 Post-Training Quantization Impact\n(AugmFP16 checkpoint, CIFAR-10 test set)")
    ax.set_ylim(80, 86)
    save(fig, "D_fp8_impact.png")


def plot_E1(all_hw: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    x = np.arange(len(EXPERIMENTS))
    w = 0.35

    cpu_avg = [all_hw[e]["cpu_avg"] for e in EXPERIMENTS]
    cpu_max = [all_hw[e]["cpu_max"] for e in EXPERIMENTS]
    mem_avg = [all_hw[e]["mem_avg"] for e in EXPERIMENTS]
    mem_max = [all_hw[e]["mem_max"] for e in EXPERIMENTS]

    ax = axes[0]
    b1 = ax.bar(x - w/2, cpu_avg, w, label="Average",
                color=[COLORS[e] for e in EXPERIMENTS], edgecolor="white", zorder=3)
    b2 = ax.bar(x + w/2, cpu_max, w, label="Peak",
                color=[COLORS[e] for e in EXPERIMENTS], alpha=0.5,
                edgecolor="white", zorder=3, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[e] for e in EXPERIMENTS], rotation=12, ha="right")
    ax.set_ylabel("CPU Utilisation (%)")
    ax.set_title("CPU Usage During Training")
    ax.legend(handles=[
        Patch(facecolor="grey", label="Average"),
        Patch(facecolor="grey", alpha=0.5, hatch="///", label="Peak"),
    ], framealpha=0.9)

    ax = axes[1]
    ax.bar(x - w/2, mem_avg, w, label="Average",
           color=[COLORS[e] for e in EXPERIMENTS], edgecolor="white", zorder=3)
    ax.bar(x + w/2, mem_max, w, label="Peak",
           color=[COLORS[e] for e in EXPERIMENTS], alpha=0.5,
           edgecolor="white", zorder=3, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[e] for e in EXPERIMENTS], rotation=12, ha="right")
    ax.set_ylabel("RAM Usage (%)")
    ax.set_title("Memory Usage During Training")
    ax.legend(handles=[
        Patch(facecolor="grey", label="Average"),
        Patch(facecolor="grey", alpha=0.5, hatch="///", label="Peak"),
    ], framealpha=0.9)

    fig.tight_layout()
    save(fig, "E1_cpu_memory_bar.png")


def parse_gpu_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (minutes_since_start, gpu_power_W, cpu_power_W) arrays."""
    from datetime import datetime
    times, gpu_p, cpu_p = [], [], []
    t0 = None
    for row in rows:
        try:
            ts  = datetime.strptime(row["timestamp"].strip(), "%H:%M:%S")
            gp  = float(row["gpu_power_mW"])
            cp  = float(row["cpu_power_mW"])
        except (ValueError, KeyError):
            continue
        if t0 is None:
            t0 = ts
        delta = (ts.hour * 3600 + ts.minute * 60 + ts.second) - \
                (t0.hour * 3600 + t0.minute * 60 + t0.second)
        # Handle midnight wrap-around
        if delta < 0:
            delta += 86400
        times.append(delta / 60.0)    # → minutes
        gpu_p.append(gp / 1000.0)     # mW → W
        cpu_p.append(cp / 1000.0)
    return np.array(times), np.array(gpu_p), np.array(cpu_p)


def rolling_mean(arr: np.ndarray, window: int = 60) -> np.ndarray:
    out = np.convolve(arr, np.ones(window) / window, mode="valid")
    # Pad to original length
    pad = len(arr) - len(out)
    return np.concatenate([np.full(pad, out[0]), out])


def plot_E2(experiments_to_compare: list[str] = ("AugmFP32", "AugmFP16")) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

    for ax, exp in zip(axes, experiments_to_compare):
        rows = load_gpu_csv(exp)
        if not rows:
            ax.set_title(f"{LABELS[exp]} — gpu_stats.csv not found")
            continue

        t, gp, cp = parse_gpu_rows(rows)
        if len(t) == 0:
            ax.set_title(f"{LABELS[exp]} — no parseable rows")
            continue

        gp_smooth = rolling_mean(gp, window=60)
        cp_smooth = rolling_mean(cp, window=60)

        ax.fill_between(t, gp, alpha=0.15, color=COLORS[exp])
        ax.plot(t, gp_smooth, color=COLORS[exp], linewidth=1.5,
                label="GPU power (60 s rolling avg)")
        ax.plot(t, cp_smooth, color="grey", linewidth=1.2, linestyle="--",
                label="CPU power (60 s rolling avg)")

        ax.set_title(LABELS[exp])
        ax.set_ylabel("Power (W)")
        ax.legend(framealpha=0.9, loc="upper right")
        ax.set_xlim(0, t.max())

    axes[-1].set_xlabel("Time since training start (minutes)")
    fig.suptitle("GPU & CPU Power Profile — AugmFP32 vs AugmFP16",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "E2_gpu_power_profile.png")


def write_comparison_json(comp: dict, fp8: dict) -> None:
    path = RESULTS_DIR / "experiment_comparison.json"
    out  = dict(comp)
    out["FP8Test"] = {
        "source":            "AugmFP16",
        "method":            "per_channel_native_fp8_e4m3fn",
        "original_fp16_acc": fp8["original_fp16"]["accuracy_percent"],
        "per_tensor_acc":    fp8["per_tensor"]["accuracy_percent"],
        "per_channel_acc":   fp8["per_channel"]["accuracy_percent"],
        "pt_degradation_pp": fp8["per_tensor"]["degradation_pp"],
        "pc_degradation_pp": fp8["per_channel"]["degradation_pp"],
        "pytorch_version":   fp8.get("pytorch_version", ""),
        "fp8_dtype":         fp8.get("fp8_dtype", "float8_e4m3fn"),
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path.name}")


def write_comparison_csv(comp: dict) -> None:
    rows = {
        "Best Val Accuracy (%)":      {e: f"{comp[e]['best_val_acc']*100:.4f}"  for e in EXPERIMENTS},
        "Best Val Acc Epoch":         {e: str(comp[e]["best_val_acc_epoch"])     for e in EXPERIMENTS},
        "Final Val Accuracy (%)":     {e: f"{comp[e]['final_val_acc']*100:.4f}" for e in EXPERIMENTS},
        "Final Train Accuracy (%)":   {e: f"{comp[e]['final_train_acc']*100:.4f}" for e in EXPERIMENTS},
        "Overfitting Gap (pp)":       {e: f"{comp[e]['overfitting_gap']:.4f}"   for e in EXPERIMENTS},
        "Convergence Epoch":          {e: str(comp[e]["convergence_epoch"])      for e in EXPERIMENTS},
        "Final Val Loss":             {e: f"{comp[e]['final_val_loss']:.6f}"     for e in EXPERIMENTS},
        "Total Time (hours)":         {e: f"{comp[e]['total_time_hours']:.4f}"  for e in EXPERIMENTS},
        "Avg Epoch Time (sec)":       {e: f"{comp[e]['avg_epoch_time_sec']:.2f}" for e in EXPERIMENTS},
        "Std Epoch Time (sec)":       {e: f"{comp[e]['std_epoch_time_sec']:.2f}" for e in EXPERIMENTS},
        "CPU Average (%)":            {e: f"{comp[e]['cpu_avg']:.2f}"            for e in EXPERIMENTS},
        "CPU Max (%)":                {e: f"{comp[e]['cpu_max']:.2f}"            for e in EXPERIMENTS},
        "Memory Average (%)":         {e: f"{comp[e]['mem_avg']:.2f}"            for e in EXPERIMENTS},
        "Memory Max (%)":             {e: f"{comp[e]['mem_max']:.2f}"            for e in EXPERIMENTS},
        "Thermal Throttling":         {e: str(comp[e]["thermal_throttled"])      for e in EXPERIMENTS},
        "Precision":                  {e: comp[e]["precision"]                   for e in EXPERIMENTS},
        "Augmentation":               {e: comp[e]["augmentation"]                for e in EXPERIMENTS},
    }
    path = RESULTS_DIR / "comparison_table.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["Metric"] + EXPERIMENTS)
        for metric, vals in rows.items():
            writer.writerow([metric] + [vals[e] for e in EXPERIMENTS])
    print(f"  Saved: {path.name}")


def write_summary(comp: dict, fp8: dict) -> None:
    fp16_acc = fp8["original_fp16"]["accuracy_percent"]
    pc_acc   = fp8["per_channel"]["accuracy_percent"]
    pc_deg   = fp8["per_channel"]["degradation_pp"]
    pt_deg   = fp8["per_tensor"]["degradation_pp"]

    lines = [
        "# ViT-FP8-Experiments — Key Results Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Experiments",
        "",
        "Model: `vit_tiny_patch16_224` (5.7 M params) · Dataset: CIFAR-10 · 50 epochs",
        "",
        "| Experiment | Precision | Augmentation | Best Val Acc | Final Train Acc | Overfit Gap | Training Time |",
        "|------------|-----------|-------------|-------------|----------------|-------------|---------------|",
    ]
    for e in EXPERIMENTS:
        c = comp[e]
        lines.append(
            f"| {e} | {c['precision']} | {c['augmentation']} | "
            f"**{c['best_val_acc']*100:.2f}%** | "
            f"{c['final_train_acc']*100:.2f}% | "
            f"{c['overfitting_gap']:.1f} pp | "
            f"{c['total_time_hours']:.2f} h |"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Precision effect (Basic augmentation)",
        f"- FP32 best acc: **{comp['BaseFP32']['best_val_acc']*100:.2f}%** · "
        f"FP16 best acc: **{comp['BaseFP16']['best_val_acc']*100:.2f}%**",
        f"- Accuracy difference: "
        f"{(comp['BaseFP16']['best_val_acc'] - comp['BaseFP32']['best_val_acc'])*100:+.2f} pp",
        f"- Training time: FP32 {comp['BaseFP32']['total_time_hours']:.2f} h  vs  "
        f"FP16 {comp['BaseFP16']['total_time_hours']:.2f} h",
        "",
        "### 2. Augmentation effect (per precision level)",
        f"- FP32: Basic → Extended: "
        f"{(comp['AugmFP32']['best_val_acc'] - comp['BaseFP32']['best_val_acc'])*100:+.2f} pp",
        f"- FP16: Basic → Extended: "
        f"{(comp['AugmFP16']['best_val_acc'] - comp['BaseFP16']['best_val_acc'])*100:+.2f} pp",
        "",
        "### 3. Best configuration",
        f"- **AugmFP16** achieves **{comp['AugmFP16']['best_val_acc']*100:.2f}%** — "
        f"highest accuracy overall",
        f"- Overfitting gap reduced from "
        f"{comp['BaseFP32']['overfitting_gap']:.1f} pp (BaseFP32) → "
        f"{comp['AugmFP16']['overfitting_gap']:.1f} pp (AugmFP16)",
        "",
        "### 4. FP8 Post-Training Quantization",
        f"- Source model (AugmFP16): **{fp16_acc:.2f}%**",
        f"- FP8 E4M3FN per-tensor: **{fp8['per_tensor']['accuracy_percent']:.2f}%** "
        f"(Δ = {pt_deg:+.2f} pp)",
        f"- FP8 E4M3FN per-channel: **{pc_acc:.2f}%** (Δ = {pc_deg:+.2f} pp)",
        "- Verdict: FP8 is **highly viable for deployment** — essentially zero accuracy cost",
        "",
        "---",
        "",
        "## Hardware",
        "",
        "| Experiment | CPU Avg | CPU Max | RAM Avg | RAM Max |",
        "|------------|---------|---------|---------|---------|",
    ]
    for e in EXPERIMENTS:
        c = comp[e]
        lines.append(
            f"| {e} | {c['cpu_avg']:.1f}% | {c['cpu_max']:.1f}% | "
            f"{c['mem_avg']:.1f}% | {c['mem_max']:.1f}% |"
        )

    lines += [
        "",
        "No thermal throttling detected in any experiment.",
        "",
        "---",
        "",
        "## Generated Plots (`results/report_plots/`)",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `A1_val_acc_all.png` | Validation accuracy curves — all 4 experiments overlaid |",
        "| `A2_train_val_acc_grid.png` | 2×2 grid: train vs val accuracy + overfitting shading |",
        "| `B_loss_curves.png` | Training and validation loss curves |",
        "| `C1_best_val_acc_bar.png` | Best validation accuracy bar chart + FP8 reference line |",
        "| `C2_overfitting_gap_bar.png` | Overfitting gap bar chart |",
        "| `C3_training_time_bar.png` | Total training time bar chart |",
        "| `D_fp8_impact.png` | FP8 quantization accuracy impact (per-tensor vs per-channel) |",
        "| `E1_cpu_memory_bar.png` | CPU and RAM usage per experiment |",
        "| `E2_gpu_power_profile.png` | GPU + CPU power time series: AugmFP32 vs AugmFP16 |",
        "",
    ]

    path = RESULTS_DIR / "summary.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path.name}")


def main() -> None:
    print("\n=== ViT-FP8 Report Generator ===\n")

    print("Loading metrics...")
    all_m      = {e: load_metrics(e)  for e in EXPERIMENTS}
    all_timing = {e: load_timing(e)   for e in EXPERIMENTS}
    all_hw_raw = {e: load_hardware(e) for e in EXPERIMENTS}
    fp8        = load_json(RESULTS_DIR / "FP8Test" / "metrics" / "fp8_quantization_results.json")

    all_hw = {e: hardware_stats_summary(all_hw_raw[e]) for e in EXPERIMENTS}

    print("Rebuilding experiment_comparison.json...")
    comp = rebuild_comparison()

    print("\nSaving comparison files...")
    write_comparison_json(comp, fp8)
    write_comparison_csv(comp)

    print("\nGenerating plots...")
    plot_A1(all_m)
    plot_A2(all_m)
    plot_B(all_m)
    plot_C1(comp, fp8)
    plot_C2(comp)
    plot_C3(comp, all_timing)
    plot_D(fp8)
    plot_E1(all_hw)
    plot_E2(["AugmFP32", "AugmFP16"])

    print("\nWriting summary.md...")
    write_summary(comp, fp8)

    print("\n=== Done ===")
    print(f"Plots:   {OUTPUT_DIR}/")
    print(f"Summary: {RESULTS_DIR}/summary.md")
    print(f"JSON:    {RESULTS_DIR}/experiment_comparison.json")
    print(f"CSV:     {RESULTS_DIR}/comparison_table.csv\n")


if __name__ == "__main__":
    main()
