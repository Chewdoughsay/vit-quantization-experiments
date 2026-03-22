"""
Hardware monitoring plots (CPU, memory, thermal, power) from training experiment data.

**Legacy** — part of the preliminary CIFAR-10 study.  Reads
``hardware_stats.json`` and ``gpu_stats.csv`` produced during training and
generates system monitoring and power consumption plots.

Usage:
    python scripts/preliminary/plot_hardware_stats.py --experiment results/preliminary/BaseFP32
    python scripts/preliminary/plot_hardware_stats.py --all
"""

import json
import csv
import argparse
import sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'lines.linewidth': 1.5,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_hardware_stats(hardware_path):
    """Load hardware stats from JSON file."""
    if not hardware_path.exists():
        return None

    try:
        with open(hardware_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_gpu_stats(gpu_path):
    """Load GPU stats from CSV file."""
    if not gpu_path.exists():
        return None

    try:
        data = {
            'timestamp': [],
            'gpu_util': [],
            'gpu_power': [],
            'cpu_power': []
        }

        with open(gpu_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data['timestamp'].append(row['timestamp'])
                data['gpu_util'].append(float(row['gpu_utilization_percent']))
                data['gpu_power'].append(float(row['gpu_power_mW']))
                data['cpu_power'].append(float(row['cpu_power_mW']))

        return data
    except (csv.Error, KeyError, ValueError, OSError):
        return None


def plot_system_monitoring(hw_stats, experiment_name, save_dir):
    """Plot CPU, memory, and thermal time-series; returns path to saved PNG or None."""
    if hw_stats is None:
        return None

    required_keys = ['cpu_percent', 'memory_percent', 'timestamps']
    if not all(key in hw_stats for key in required_keys):
        print(f"  ⚠️  Missing required data in hardware_stats.json for {experiment_name}")
        return None

    timestamps = np.array(hw_stats['timestamps']) / 60  # Convert to minutes
    cpu_percent = hw_stats['cpu_percent']
    mem_percent = hw_stats['memory_percent']
    thermal = hw_stats.get('thermal_pressure', [])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{experiment_name} - System Monitoring', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(timestamps, cpu_percent, color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax1.fill_between(timestamps, cpu_percent, alpha=0.3, color='#2E86AB')
    ax1.set_ylabel('CPU Usage (%)', fontweight='bold')
    ax1.set_xlabel('Time (minutes)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(100, max(cpu_percent) * 1.1))

    avg_cpu = np.mean(cpu_percent)
    max_cpu = np.max(cpu_percent)
    ax1.axhline(y=avg_cpu, color='red', linestyle='--', alpha=0.6,
                label=f'Avg: {avg_cpu:.1f}%', linewidth=1.5)
    ax1.legend(loc='upper right')
    ax1.text(0.02, 0.98, f'Max: {max_cpu:.1f}%', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]
    ax2.plot(timestamps, mem_percent, color='#A23B72', linewidth=1.5, alpha=0.8)
    ax2.fill_between(timestamps, mem_percent, alpha=0.3, color='#A23B72')
    ax2.set_ylabel('Memory Usage (%)', fontweight='bold')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    avg_mem = np.mean(mem_percent)
    max_mem = np.max(mem_percent)
    ax2.axhline(y=avg_mem, color='red', linestyle='--', alpha=0.6,
                label=f'Avg: {avg_mem:.1f}%', linewidth=1.5)
    ax2.legend(loc='upper right')
    ax2.text(0.02, 0.98, f'Max: {max_mem:.1f}%', transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax3 = axes[2]
    if thermal and any(t > 0 for t in thermal):
        ax3.plot(timestamps, thermal, color='#F18F01', linewidth=1.5, alpha=0.8)
        ax3.fill_between(timestamps, thermal, alpha=0.3, color='#F18F01')
        max_thermal = np.max(thermal)
        ax3.text(0.02, 0.98, f'Max: {max_thermal}', transform=ax3.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    else:
        ax3.text(0.5, 0.5, 'No Thermal Throttling Detected ✓',
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax3.set_ylabel('Thermal Level', fontweight='bold')
    ax3.set_xlabel('Time (minutes)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f'{experiment_name}_hardware_monitoring.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def plot_power_consumption(gpu_stats, experiment_name, save_dir):
    """Plot GPU and CPU power consumption from powermetrics data; returns path to saved PNG or None."""
    if gpu_stats is None:
        return None

    n_samples = len(gpu_stats['gpu_power'])
    time_minutes = np.arange(n_samples) / 60  # Assuming 1 sample per second
    gpu_power_w = np.array(gpu_stats['gpu_power']) / 1000
    cpu_power_w = np.array(gpu_stats['cpu_power']) / 1000
    total_power_w = gpu_power_w + cpu_power_w

    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle(f'{experiment_name} - Power Consumption (Apple Silicon)', fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(time_minutes, gpu_power_w, color='#D62828', linewidth=1.5, alpha=0.8, label='GPU Power')
    ax1.plot(time_minutes, cpu_power_w, color='#2E86AB', linewidth=1.5, alpha=0.8, label='CPU Power')
    ax1.fill_between(time_minutes, gpu_power_w, alpha=0.2, color='#D62828')
    ax1.fill_between(time_minutes, cpu_power_w, alpha=0.2, color='#2E86AB')
    ax1.set_ylabel('Power (W)', fontweight='bold')
    ax1.set_xlabel('Time (minutes)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    avg_gpu_power = np.mean(gpu_power_w)
    avg_cpu_power = np.mean(cpu_power_w)
    ax1.text(0.02, 0.98, f'GPU Avg: {avg_gpu_power:.1f}W | CPU Avg: {avg_cpu_power:.1f}W',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2 = axes[1]
    ax2.plot(time_minutes, total_power_w, color='#6A4C93', linewidth=2, alpha=0.8)
    ax2.fill_between(time_minutes, total_power_w, alpha=0.3, color='#6A4C93')
    ax2.set_ylabel('Total Power (W)', fontweight='bold')
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, alpha=0.3)

    avg_total = np.mean(total_power_w)
    max_total = np.max(total_power_w)
    ax2.axhline(y=avg_total, color='red', linestyle='--', alpha=0.6,
                label=f'Avg: {avg_total:.1f}W', linewidth=1.5)
    ax2.legend(loc='upper right')
    ax2.text(0.02, 0.98, f'Max: {max_total:.1f}W | Avg: {avg_total:.1f}W',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f'{experiment_name}_power_consumption.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def plot_experiment(exp_path):
    """Plot hardware stats for a single experiment."""
    exp_path = Path(exp_path)
    exp_name = exp_path.name

    print(f"\nGenerating hardware plots for: {exp_name}")

    hardware_file = exp_path / 'metrics' / 'hardware_stats.json'
    gpu_file = exp_path / 'metrics' / 'gpu_stats.csv'
    output_dir = exp_path / 'plots'

    plots_generated = 0

    print("  Loading system monitoring data...")
    hw_stats = load_hardware_stats(hardware_file)
    if hw_stats:
        print("  Creating system monitoring plot...")
        hw_path = plot_system_monitoring(hw_stats, exp_name, output_dir)
        if hw_path:
            print(f"  ✓ Saved: {hw_path}")
            plots_generated += 1
    else:
        print("  ⚠️  No hardware_stats.json found")

    print("  Loading power consumption data...")
    gpu_stats = load_gpu_stats(gpu_file)
    if gpu_stats:
        print("  Creating power consumption plot...")
        power_path = plot_power_consumption(gpu_stats, exp_name, output_dir)
        if power_path:
            print(f"  ✓ Saved: {power_path}")
            plots_generated += 1
    else:
        print("  ⚠️  No gpu_stats.csv found (experiments without sudo don't have power data)")

    if plots_generated > 0:
        print(f"✓ Generated {plots_generated} hardware plot(s) for {exp_name}")
    else:
        print(f"⚠️  No hardware plots generated for {exp_name}")

    return plots_generated


def main():
    parser = argparse.ArgumentParser(
        description='Generate hardware monitoring plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single experiment
  python scripts/plot_hardware_stats.py --experiment results/preliminary/BaseFP32

  # Plot all experiments
  python scripts/plot_hardware_stats.py --all

  # Plot specific experiments
  python scripts/plot_hardware_stats.py --experiments results/preliminary/BaseFP32 results/preliminary/AugmFP16
        """
    )

    parser.add_argument('--experiment', type=str, help='Path to single experiment directory')
    parser.add_argument('--experiments', nargs='+', help='List of experiment directories')
    parser.add_argument('--all', action='store_true', help='Plot all experiments in results/')

    args = parser.parse_args()

    experiments = []

    if args.experiment:
        experiments.append(args.experiment)
    elif args.experiments:
        experiments.extend(args.experiments)
    elif args.all:
        results_dir = Path('results/preliminary')
        if results_dir.exists():
            for exp_dir in results_dir.iterdir():
                if exp_dir.is_dir() and (exp_dir / 'metrics').exists():
                    experiments.append(str(exp_dir))

        if experiments:
            print(f"Found {len(experiments)} experiments: {[Path(e).name for e in experiments]}")
        else:
            print("Error: No experiments found in results/ directory")
            sys.exit(1)
    else:
        print("Error: Must specify --experiment, --experiments, or --all")
        parser.print_help()
        sys.exit(1)

    total_plots = 0
    for exp in experiments:
        plots_count = plot_experiment(exp)
        total_plots += plots_count

    print(f"\n{'='*60}")
    print(f"✓ All done! Generated {total_plots} hardware monitoring plots")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
