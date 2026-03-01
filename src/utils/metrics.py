"""Metrics tracking, accuracy calculation, and timing utilities for ViT training."""
import time
import json
from pathlib import Path
import torch


class MetricsTracker:
    """Tracks and persists per-epoch training metrics (loss, accuracy, LR, timing) to JSON."""

    def __init__(self, save_dir='results/metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.reset()

    def reset(self):
        """Clear all stored metrics."""
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': [],
            'learning_rates': [],
        }
        self.current_epoch = 0

    def update(self, **kwargs):
        """Append epoch metric values; unknown keys are silently ignored."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_best_acc(self):
        """Return the best validation accuracy, or 0.0 if none recorded."""
        if self.metrics['val_acc']:
            return max(self.metrics['val_acc'])
        return 0.0

    def save(self, filename='metrics.json'):
        """Save metrics to a JSON file in save_dir."""
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")

    def load(self, filename='metrics.json'):
        """Load metrics from a JSON file in save_dir."""
        load_path = self.save_dir / filename
        with open(load_path, 'r') as f:
            self.metrics = json.load(f)
        print(f"Metrics loaded from {load_path}")


def calculate_accuracy(outputs, targets):
    """Return fraction of correct predictions (argmax of outputs vs targets)."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


class Timer:
    """Context manager that prints elapsed time for a named code block."""

    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.name} took {self.elapsed:.2f} seconds")


if __name__ == '__main__':
    # Test
    tracker = MetricsTracker(save_dir='test_metrics')
    tracker.update(
        train_loss=0.5,
        train_acc=0.85,
        val_loss=0.6,
        val_acc=0.82,
        epoch_time=120.5
    )
    tracker.save('test.json')
    print(f"Best accuracy: {tracker.get_best_acc()}")