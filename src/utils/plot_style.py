"""
Shared plot style, color palette, and save helper for all report plots.

Usage:
    from src.utils.plot_style import apply_style, COLORS, save_fig
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAVE_DPI = 300

COLORS = {
    "FP32":    "#0072B2",
    "FP16":    "#009E73",
    "INT8":    "#E69F00",
    "INT8-pt": "#E69F00",
    "INT8-pc": "#D55E00",
}


def apply_style() -> None:
    """Apply the project-wide matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":       "DejaVu Serif",
        "font.size":         12,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    13,
        "legend.fontsize":   11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F8F8F8",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.4,
        "grid.linestyle":    "--",
    })


def save_fig(fig: plt.Figure, output_dir: str | Path, name: str) -> Path:
    """Save a figure to *output_dir/name*, creating dirs as needed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
