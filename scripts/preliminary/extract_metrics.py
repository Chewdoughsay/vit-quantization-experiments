"""
Load, analyse, and compare training metrics from ViT experiment directories.

**Legacy** — part of the preliminary CIFAR-10 study.  Reads
``final_metrics.json`` files produced by ``train.py`` and prints per-
experiment statistics or a side-by-side comparison table.

Usage:
    python scripts/preliminary/extract_metrics.py --experiment results/preliminary/BaseFP32
    python scripts/preliminary/extract_metrics.py --compare results/preliminary/BaseFP32 results/preliminary/AugmFP16
    python scripts/preliminary/extract_metrics.py --all --output metrics_summary.csv
"""

import json
import argparse
import sys
from pathlib import Path
import numpy as np


def load_experiment_metrics(metrics_path):
    """Load and return metrics dict from a final_metrics.json file."""
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def compute_statistics(metrics):
    """Compute best/final accuracy, timing, convergence epoch, and overfitting score from metrics dict."""
    train_loss = metrics.get('train_loss', [])
    train_acc = metrics.get('train_acc', [])
    val_loss = metrics.get('val_loss', [])
    val_acc = metrics.get('val_acc', [])
    epoch_time = metrics.get('epoch_time', [])

    stats = {}

    if val_acc:
        stats['best_val_acc'] = max(val_acc)
        stats['best_val_acc_epoch'] = val_acc.index(stats['best_val_acc']) + 1
    else:
        stats['best_val_acc'] = 0.0
        stats['best_val_acc_epoch'] = 0

    stats['final_train_loss'] = train_loss[-1] if train_loss else 0.0
    stats['final_val_loss'] = val_loss[-1] if val_loss else 0.0
    stats['final_train_acc'] = train_acc[-1] if train_acc else 0.0
    stats['final_val_acc'] = val_acc[-1] if val_acc else 0.0

    if epoch_time:
        stats['avg_epoch_time'] = np.mean(epoch_time)
        stats['total_time_minutes'] = sum(epoch_time) / 60
    else:
        stats['avg_epoch_time'] = 0.0
        stats['total_time_minutes'] = 0.0

    if val_acc:
        threshold = stats['best_val_acc'] * 0.995  # 99.5% of best
        convergence_epochs = [i+1 for i, acc in enumerate(val_acc) if acc >= threshold]
        stats['convergence_epoch'] = convergence_epochs[0] if convergence_epochs else len(val_acc)
    else:
        stats['convergence_epoch'] = 0

    stats['generalization_gap'] = stats['final_train_acc'] - stats['final_val_acc']
    stats['overfitting_score'] = stats['best_val_acc'] - stats['final_val_acc']

    return stats


def print_experiment_summary(experiment_name, stats, detailed=False):
    """Print a formatted summary table for a single experiment."""
    print(f"\nExperiment Summary: {experiment_name}")
    print("="*60)

    if detailed:
        # Detailed view with sections
        print("Accuracy Metrics:")
        print(f"  Best Validation Accuracy: {stats['best_val_acc']:.4f} (epoch {stats['best_val_acc_epoch']})")
        print(f"  Final Training Accuracy: {stats['final_train_acc']:.4f}")
        print(f"  Final Validation Accuracy: {stats['final_val_acc']:.4f}")
        print(f"  Generalization Gap: {stats['generalization_gap']:.4f} (train - val)")

        print("\nLoss Metrics:")
        print(f"  Final Training Loss: {stats['final_train_loss']:.4f}")
        print(f"  Final Validation Loss: {stats['final_val_loss']:.4f}")

        print("\nTraining Dynamics:")
        print(f"  Convergence Epoch: {stats['convergence_epoch']}")
        print(f"  Overfitting Score: {stats['overfitting_score']:.4f} (best - final val_acc)")

        print("\nTiming:")
        print(f"  Average Epoch Time: {stats['avg_epoch_time']:.1f} seconds")
        print(f"  Total Training Time: {stats['total_time_minutes']:.1f} minutes")
    else:
        # Compact view
        print(f"Best Validation Accuracy: {stats['best_val_acc']:.4f} (epoch {stats['best_val_acc_epoch']})")
        print(f"Final Training Loss: {stats['final_train_loss']:.4f}")
        print(f"Final Validation Loss: {stats['final_val_loss']:.4f}")
        print(f"Convergence Epoch: {stats['convergence_epoch']}")
        print(f"Average Epoch Time: {stats['avg_epoch_time']:.1f} seconds")
        print(f"Total Training Time: {stats['total_time_minutes']:.1f} minutes")

    print("="*60 + "\n")


def compare_experiments(experiment_paths, output_csv=None):
    """Print a side-by-side comparison table; optionally save to CSV."""
    results = []

    for exp_path in experiment_paths:
        exp_path = Path(exp_path)
        exp_name = exp_path.name
        metrics_file = exp_path / 'metrics' / 'final_metrics.json'

        try:
            metrics = load_experiment_metrics(metrics_file)
            stats = compute_statistics(metrics)
            results.append({'name': exp_name, 'stats': stats})
        except FileNotFoundError:
            print(f"Warning: Metrics not found for {exp_name}, skipping...")
            continue

    if not results:
        print("No experiments found with valid metrics")
        return results

    print("\nExperiment Comparison")
    print("="*80)
    print(f"{'Experiment':<16} | {'Best Acc':<8} | {'Final Loss':<10} | {'Conv Epoch':<10} | {'Time (min)':<10}")
    print("-"*80)

    for result in results:
        name = result['name']
        stats = result['stats']
        print(f"{name:<16} | {stats['best_val_acc']:>8.4f} | "
              f"{stats['final_val_loss']:>10.4f} | {stats['convergence_epoch']:>10} | "
              f"{stats['total_time_minutes']:>10.1f}")

    print("="*80 + "\n")

    if output_csv:
        import csv
        output_path = Path(output_csv)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Experiment', 'Best_Val_Acc', 'Best_Acc_Epoch',
                'Final_Train_Loss', 'Final_Val_Loss',
                'Final_Train_Acc', 'Final_Val_Acc',
                'Convergence_Epoch', 'Generalization_Gap', 'Overfitting_Score',
                'Avg_Epoch_Time_sec', 'Total_Time_min'
            ])

            for result in results:
                name = result['name']
                stats = result['stats']
                writer.writerow([
                    name,
                    stats['best_val_acc'],
                    stats['best_val_acc_epoch'],
                    stats['final_train_loss'],
                    stats['final_val_loss'],
                    stats['final_train_acc'],
                    stats['final_val_acc'],
                    stats['convergence_epoch'],
                    stats['generalization_gap'],
                    stats['overfitting_score'],
                    stats['avg_epoch_time'],
                    stats['total_time_minutes']
                ])

        print(f"Comparison table saved to: {output_path}\n")

    return results


def discover_experiments(results_dir='results/preliminary'):
    """Return sorted list of experiment directories containing final_metrics.json."""
    results_path = Path(results_dir)

    if not results_path.exists():
        return []

    experiments = []

    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir():
            metrics_file = exp_dir / 'metrics' / 'final_metrics.json'
            if metrics_file.exists():
                experiments.append(exp_dir)

    return sorted(experiments)


def main():
    parser = argparse.ArgumentParser(
        description='Extract and analyze training metrics from experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python scripts/extract_metrics.py --experiment results/preliminary/BaseFP32

  # Single experiment with detailed stats
  python scripts/extract_metrics.py --experiment results/preliminary/BaseFP32 --detailed

  # Compare multiple experiments
  python scripts/extract_metrics.py --compare results/preliminary/BaseFP32 results/preliminary/BaseFP16 results/preliminary/AugmFP16

  # Extract all experiments and save to CSV
  python scripts/extract_metrics.py --all --output metrics_summary.csv
        """
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Path to single experiment directory (e.g., results/preliminary/BaseFP32)'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='List of experiment directories to compare'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all experiments in results/ directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for comparison results'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed statistics (for single experiment)'
    )

    args = parser.parse_args()

    # Single experiment mode
    if args.experiment:
        exp_path = Path(args.experiment)
        metrics_file = exp_path / 'metrics' / 'final_metrics.json'

        try:
            print(f"Loading metrics from: {metrics_file}")
            metrics = load_experiment_metrics(metrics_file)
            stats = compute_statistics(metrics)
            print_experiment_summary(exp_path.name, stats, detailed=args.detailed)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing metrics JSON: {e}")
            sys.exit(1)

    # Comparison mode
    elif args.compare:
        print(f"Comparing {len(args.compare)} experiments...")
        compare_experiments(args.compare, output_csv=args.output)

    # All experiments mode
    elif args.all:
        print("Discovering experiments in results/ directory...")
        experiments = discover_experiments('results/preliminary')

        if not experiments:
            print("No experiments found in results/ directory")
            sys.exit(1)

        print(f"Found {len(experiments)} experiments: {[e.name for e in experiments]}")
        compare_experiments(experiments, output_csv=args.output)

    else:
        print("Error: Must specify --experiment, --compare, or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
