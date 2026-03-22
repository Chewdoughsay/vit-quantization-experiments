"""
Compare all four preliminary ViT experiments (performance, timing, hardware).

**Legacy** — part of the preliminary CIFAR-10 study.  Loads metrics from
BaseFP32 / AugmFP32 / BaseFP16 / AugmFP16 and outputs a comparison table
plus a detailed JSON file for downstream plotting.

Outputs:
    results/comparison_table.csv
    results/experiment_comparison.json

Usage:
    python scripts/preliminary/compare_experiments.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import csv


def load_experiment_data(experiment_name):
    """Load metrics, timing, and hardware JSON files for an experiment; returns dict with 'exists' flag."""
    exp_path = Path('results/preliminary') / experiment_name
    metrics_dir = exp_path / 'metrics'

    if not exp_path.exists():
        return {'exists': False, 'name': experiment_name}

    data = {
        'exists': True,
        'name': experiment_name,
        'metrics': None,
        'timing': None,
        'hardware': None
    }

    metrics_file = metrics_dir / 'final_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            data['metrics'] = json.load(f)

    timing_file = metrics_dir / 'timing_report.json'
    if timing_file.exists():
        with open(timing_file, 'r') as f:
            data['timing'] = json.load(f)

    hardware_file = metrics_dir / 'hardware_stats.json'
    if hardware_file.exists():
        with open(hardware_file, 'r') as f:
            data['hardware'] = json.load(f)

    return data


def compute_comprehensive_metrics(data):
    """Compute performance, timing, and hardware metrics from experiment data dict."""
    if not data['exists'] or not data['metrics']:
        return None

    metrics = data['metrics']
    timing = data.get('timing', {})
    hardware = data.get('hardware', {})

    train_loss = metrics.get('train_loss', [])
    val_loss = metrics.get('val_loss', [])
    train_acc = metrics.get('train_acc', [])
    val_acc = metrics.get('val_acc', [])
    epoch_times = metrics.get('epoch_time', [])

    best_val_acc = max(val_acc) if val_acc else 0.0
    best_val_acc_epoch = val_acc.index(best_val_acc) + 1 if val_acc else 0
    final_val_acc = val_acc[-1] if val_acc else 0.0
    final_train_acc = train_acc[-1] if train_acc else 0.0
    final_val_loss = val_loss[-1] if val_loss else 0.0
    final_train_loss = train_loss[-1] if train_loss else 0.0

    overfitting_gap = (final_train_acc - final_val_acc) * 100
    overfitting_score = best_val_acc - final_val_acc

    # First epoch reaching 99.5% of best accuracy
    threshold = best_val_acc * 0.995
    convergence_epochs = [i+1 for i, acc in enumerate(val_acc) if acc >= threshold]
    convergence_epoch = convergence_epochs[0] if convergence_epochs else len(val_acc)

    num_epochs = len(train_loss)
    total_time_hours = timing.get('total_duration_hours', 0.0)
    total_time_minutes = timing.get('total_duration_minutes', 0.0)

    if epoch_times:
        avg_epoch_time = np.mean(epoch_times)
        std_epoch_time = np.std(epoch_times)
        min_epoch_time = np.min(epoch_times)
        max_epoch_time = np.max(epoch_times)
    else:
        avg_epoch_time = timing.get('avg_epoch_time_seconds', 0.0)
        std_epoch_time = 0.0
        min_epoch_time = avg_epoch_time
        max_epoch_time = avg_epoch_time

    hw_summary = hardware.get('summary', {})
    cpu_avg = hw_summary.get('avg_cpu', 0.0)
    cpu_max = hw_summary.get('max_cpu', 0.0)
    mem_avg = hw_summary.get('avg_mem', 0.0)
    thermal_max = hw_summary.get('max_thermal', 0)
    thermal_throttled = hw_summary.get('throttled', False)

    result = {
        'num_epochs': num_epochs,
        'best_val_acc': best_val_acc,
        'best_val_acc_epoch': best_val_acc_epoch,
        'final_val_acc': final_val_acc,
        'final_train_acc': final_train_acc,
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'overfitting_gap': overfitting_gap,
        'overfitting_score': overfitting_score,
        'convergence_epoch': convergence_epoch,
        'total_time_hours': total_time_hours,
        'total_time_minutes': total_time_minutes,
        'avg_epoch_time_sec': avg_epoch_time,
        'std_epoch_time_sec': std_epoch_time,
        'min_epoch_time_sec': min_epoch_time,
        'max_epoch_time_sec': max_epoch_time,
        'cpu_avg': cpu_avg,
        'cpu_max': cpu_max,
        'mem_avg': mem_avg,
        'thermal_max': thermal_max,
        'thermal_throttled': thermal_throttled,
        'precision': 'FP16 (AMP)' if 'FP16' in data['name'] else 'FP32',
        'augmentation': 'Extended' if 'Augm' in data['name'] else 'Basic'
    }

    return result


def print_comparison_table(experiments_metrics):
    """Print formatted comparison table to console."""
    print("\n" + "="*100)
    print("COMPREHENSIVE EXPERIMENT COMPARISON")
    print("="*100 + "\n")

    exp_order = ['BaseFP32', 'AugmFP32', 'BaseFP16', 'AugmFP16']

    print("PERFORMANCE METRICS")
    print("-"*100)
    print(f"{'Metric':<30} | {'BaseFP32':<15} | {'AugmFP32':<15} | {'BaseFP16':<15} | {'AugmFP16':<15}")
    print("-"*100)

    metrics_to_show = [
        ('Epochs Trained', 'num_epochs', '{}'),
        ('Best Val Accuracy', 'best_val_acc', '{:.2%}'),
        ('Final Val Accuracy', 'final_val_acc', '{:.2%}'),
        ('Final Train Accuracy', 'final_train_acc', '{:.2%}'),
        ('Overfitting Gap', 'overfitting_gap', '{:.2f}%'),
        ('Final Val Loss', 'final_val_loss', '{:.4f}'),
        ('Convergence Epoch', 'convergence_epoch', '{}'),
    ]

    for label, key, fmt in metrics_to_show:
        row = [label]
        for exp_name in exp_order:
            if exp_name in experiments_metrics and experiments_metrics[exp_name]:
                value = experiments_metrics[exp_name].get(key, 0)
                row.append(fmt.format(value))
            else:
                row.append('NOT RUN')
        print(f"{row[0]:<30} | {row[1]:<15} | {row[2]:<15} | {row[3]:<15} | {row[4]:<15}")

    # Timing Metrics
    print("\n" + "-"*100)
    print("TIMING METRICS")
    print("-"*100)

    timing_metrics = [
        ('Total Time (hours)', 'total_time_hours', '{:.2f}h'),
        ('Time/Epoch (sec)', 'avg_epoch_time_sec', '{:.1f}s'),
        ('Time/Epoch StdDev', 'std_epoch_time_sec', '±{:.1f}s'),
    ]

    for label, key, fmt in timing_metrics:
        row = [label]
        for exp_name in exp_order:
            if exp_name in experiments_metrics and experiments_metrics[exp_name]:
                value = experiments_metrics[exp_name].get(key, 0)
                row.append(fmt.format(value))
            else:
                row.append('NOT RUN')
        print(f"{row[0]:<30} | {row[1]:<15} | {row[2]:<15} | {row[3]:<15} | {row[4]:<15}")

    # Hardware Metrics
    print("\n" + "-"*100)
    print("HARDWARE UTILIZATION")
    print("-"*100)

    hardware_metrics = [
        ('CPU Average', 'cpu_avg', '{:.1f}%'),
        ('CPU Max', 'cpu_max', '{:.1f}%'),
        ('Memory Average', 'mem_avg', '{:.1f}%'),
        ('Thermal Throttling', 'thermal_throttled', '{}'),
    ]

    for label, key, fmt in hardware_metrics:
        row = [label]
        for exp_name in exp_order:
            if exp_name in experiments_metrics and experiments_metrics[exp_name]:
                value = experiments_metrics[exp_name].get(key, 0)
                if key == 'thermal_throttled':
                    value = 'Yes' if value else 'No'
                row.append(fmt.format(value))
            else:
                row.append('NOT RUN')
        print(f"{row[0]:<30} | {row[1]:<15} | {row[2]:<15} | {row[3]:<15} | {row[4]:<15}")

    print("="*100 + "\n")


def print_analysis_summary(experiments_metrics):
    """Print analysis summary with key insights."""
    print("\n" + "="*100)
    print("ANALYSIS SUMMARY")
    print("="*100 + "\n")

    best_acc = max(m['best_val_acc'] for m in experiments_metrics.values() if m)
    best_exp = [name for name, m in experiments_metrics.items()
                if m and m['best_val_acc'] == best_acc][0]

    valid_overfit = {name: m['overfitting_gap']
                     for name, m in experiments_metrics.items() if m}
    best_overfit = min(valid_overfit.values())
    best_overfit_exp = [name for name, gap in valid_overfit.items()
                        if gap == best_overfit][0]

    valid_times = {name: m['total_time_hours']
                   for name, m in experiments_metrics.items() if m}
    fastest_time = min(valid_times.values())
    fastest_exp = [name for name, t in valid_times.items()
                   if t == fastest_time][0]

    print(f"Best Accuracy: {best_exp} ({best_acc:.2%})")
    print(f"Best Generalization (lowest overfitting): {best_overfit_exp} ({best_overfit:.2f}%)")
    print(f"Fastest Training: {fastest_exp} ({fastest_time:.2f} hours)")

    if 'BaseFP32' in experiments_metrics and 'BaseFP16' in experiments_metrics:
        if experiments_metrics['BaseFP32'] and experiments_metrics['BaseFP16']:
            fp32_time = experiments_metrics['BaseFP32']['total_time_hours']
            fp16_time = experiments_metrics['BaseFP16']['total_time_hours']
            speedup = fp32_time / fp16_time
            reduction = ((fp32_time - fp16_time) / fp32_time) * 100
            print(f"\nFP16 Speedup (BaseFP16 vs BaseFP32): {speedup:.2f}x faster ({reduction:.0f}% time reduction)")

    if 'AugmFP32' in experiments_metrics and 'AugmFP16' in experiments_metrics:
        if experiments_metrics['AugmFP32'] and experiments_metrics['AugmFP16']:
            fp32_time = experiments_metrics['AugmFP32']['total_time_hours']
            fp16_time = experiments_metrics['AugmFP16']['total_time_hours']
            speedup = fp32_time / fp16_time
            reduction = ((fp32_time - fp16_time) / fp32_time) * 100
            print(f"FP16 Speedup (AugmFP16 vs AugmFP32): {speedup:.2f}x faster ({reduction:.0f}% time reduction)")

    print(f"\nRecommended: AugmFP16")
    if 'AugmFP16' in experiments_metrics and experiments_metrics['AugmFP16']:
        augm = experiments_metrics['AugmFP16']
        print(f"   - Accuracy: {augm['best_val_acc']:.2%}")
        print(f"   - Overfitting: {augm['overfitting_gap']:.2f}%")
        print(f"   - Training Time: {augm['total_time_hours']:.2f} hours")
        print(f"   - Best balance of speed, accuracy, and generalization!")

    print("="*100 + "\n")


def save_comparison_csv(experiments_metrics, output_path='results/preliminary/comparison_table.csv'):
    """Save comparison table to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp_order = ['BaseFP32', 'AugmFP32', 'BaseFP16', 'AugmFP16']

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['Metric', 'BaseFP32', 'AugmFP32', 'BaseFP16', 'AugmFP16'])

        all_metrics = [
            ('Epochs Trained', 'num_epochs'),
            ('Best Val Accuracy (%)', 'best_val_acc', 100),
            ('Final Val Accuracy (%)', 'final_val_acc', 100),
            ('Final Train Accuracy (%)', 'final_train_acc', 100),
            ('Overfitting Gap (%)', 'overfitting_gap', 1),
            ('Final Val Loss', 'final_val_loss', 1),
            ('Total Time (hours)', 'total_time_hours', 1),
            ('Time/Epoch (sec)', 'avg_epoch_time_sec', 1),
            ('Time/Epoch StdDev', 'std_epoch_time_sec', 1),
            ('CPU Average (%)', 'cpu_avg', 1),
            ('CPU Max (%)', 'cpu_max', 1),
            ('Memory Average (%)', 'mem_avg', 1),
            ('Thermal Throttling', 'thermal_throttled', None),
        ]

        for metric_info in all_metrics:
            label = metric_info[0]
            key = metric_info[1]
            scale = metric_info[2] if len(metric_info) > 2 else 1

            row = [label]
            for exp_name in exp_order:
                if exp_name in experiments_metrics and experiments_metrics[exp_name]:
                    value = experiments_metrics[exp_name].get(key, 0)
                    if key == 'thermal_throttled':
                        value = 'Yes' if value else 'No'
                    elif scale:
                        value = value * scale
                    row.append(value)
                else:
                    row.append('NOT RUN')
            writer.writerow(row)

    print(f"Comparison table saved to: {output_path}")


def save_detailed_json(experiments_metrics, output_path='results/preliminary/experiment_comparison.json'):
    """Save detailed metrics to JSON for plotting scripts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(experiments_metrics, f, indent=2)

    print(f"Detailed metrics saved to: {output_path}")


def main():
    print("\n" + "="*100)
    print("LOADING EXPERIMENT DATA...")
    print("="*100 + "\n")

    experiments = ['BaseFP32', 'AugmFP32', 'BaseFP16', 'AugmFP16']
    experiments_data = {}
    experiments_metrics = {}

    for exp_name in experiments:
        print(f"Loading {exp_name}...", end=' ')
        data = load_experiment_data(exp_name)

        if data['exists']:
            experiments_data[exp_name] = data
            metrics = compute_comprehensive_metrics(data)
            if metrics:
                experiments_metrics[exp_name] = metrics
                print("✓")
            else:
                print("(incomplete data)")
        else:
            print("(not found)")

    print()

    if not experiments_metrics:
        print("No experiments found! Run experiments first:")
        print("  $ python scripts/train_BaseFP32.py")
        print("  $ python scripts/train_AugmFP32.py")
        print("  $ python scripts/train_BaseFP16.py")
        print("  $ python scripts/train_AugmFP16.py")
        sys.exit(1)

    print_comparison_table(experiments_metrics)
    print_analysis_summary(experiments_metrics)

    print("SAVING OUTPUTS...")
    print("-"*100)
    save_comparison_csv(experiments_metrics, 'results/preliminary/comparison_table.csv')
    save_detailed_json(experiments_metrics, 'results/preliminary/experiment_comparison.json')
    print("="*100 + "\n")

    print("Analysis complete! Use the generated files for your report:")
    print("   - results/preliminary/comparison_table.csv (for LaTeX tables)")
    print("   - results/preliminary/experiment_comparison.json (for plotting)")
    print()


if __name__ == '__main__':
    main()
