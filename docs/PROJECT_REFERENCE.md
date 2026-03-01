# ViT-FP8-Experiments — Project Reference

This document describes the purpose of every file in the project and the full set of metrics that are collected, extracted, and compared across experiments.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Experimental Design](#2-experimental-design)
3. [Directory Structure](#3-directory-structure)
4. [Configuration Files (`configs/`)](#4-configuration-files-configs)
5. [Source Modules (`src/`)](#5-source-modules-src)
6. [Scripts (`scripts/`)](#6-scripts-scripts)
7. [Results Layout (`results/`)](#7-results-layout-results)
8. [Metrics Reference](#8-metrics-reference)
   - [Per-Epoch Training Metrics](#per-epoch-training-metrics)
   - [Derived Summary Metrics](#derived-summary-metrics)
   - [Timing Metrics](#timing-metrics)
   - [Hardware Utilization Metrics](#hardware-utilization-metrics)
   - [FP8 Quantization Metrics](#fp8-quantization-metrics)
9. [What Is Being Compared](#9-what-is-being-compared)

---

## 1. Project Goal

The project systematically measures the trade-offs between **numerical precision** and **model accuracy** when training a Vision Transformer (ViT-Tiny) on CIFAR-10. Three precision levels are studied:

| Precision | Description |
|-----------|-------------|
| **FP32** | Standard full precision — reference baseline |
| **FP16** | Mixed precision via PyTorch AMP — faster training |
| **FP8** | Post-training quantization (E4M3) — inference efficiency test |

Each precision level is crossed with two data augmentation strategies (basic vs. extended), yielding a 2×2 factorial design plus one FP8 post-training quantization evaluation.

**Core research questions:**
- How much accuracy is lost going FP32 → FP16 → FP8?
- Does extended augmentation close any precision-induced accuracy gap?
- How much wall-clock time and hardware pressure does each configuration require?
- Which configuration offers the best speed / accuracy trade-off for deployment?

---

## 2. Experimental Design

| Experiment | Precision | Augmentation | Regularization | Notes |
|------------|-----------|-------------|----------------|-------|
| **BaseFP32** | FP32 | Basic | None | Baseline reference |
| **BaseFP16** | FP16 (AMP) | Basic | None | Speedup test |
| **AugmFP32** | FP32 | Extended | Full | Best FP32 generalization |
| **AugmFP16** | FP16 (AMP) | Extended | Full | Best overall trade-off |
| **FP8Test** | FP8 (simulated) | Extended | N/A | Post-training quantization only |

**Shared settings across all training runs:**
- Model: `vit_tiny_patch16_224` (5.7 M parameters)
- Dataset: CIFAR-10 (50 K train / 10 K test images)
- Optimizer: AdamW, initial LR = 0.001, cosine annealing schedule
- Loss: CrossEntropyLoss (+ optional label smoothing)
- Batch size: 128
- Epochs: 50

**Basic augmentation:** RandomCrop(224, padding=28) + RandomHorizontalFlip
**Extended augmentation:** Basic + ColorJitter + RandomRotation(±15°) + RandomErasing
**Full regularization:** label smoothing = 0.1, gradient clipping = 1.0, LR warmup = 5 epochs

---

## 3. Directory Structure

```
ViT-FP8-experiments/
│
├── configs/                     # YAML experiment configurations
│   ├── BaseFP32.yaml
│   ├── BaseFP16.yaml
│   ├── AugmFP32.yaml
│   ├── AugmFP16.yaml
│   └── FP8Test.yaml
│
├── src/                         # Importable Python modules
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   └── vit_model.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       ├── metrics.py
│       ├── system_monitor.py
│       └── gpu_monitor.py
│
├── scripts/                     # Executable entry points
│   ├── train.py
│   ├── evaluate_fp8_quantization.py
│   ├── extract_metrics.py
│   ├── generate_plots.py
│   ├── compare_experiments.py
│   └── plot_hardware_stats.py
│
├── results/                     # All outputs (auto-generated)
│   ├── BaseFP32/
│   ├── BaseFP16/
│   ├── AugmFP32/
│   ├── AugmFP16/
│   ├── FP8Test/
│   ├── comparison_plots/
│   └── experiment_comparison.json
│
├── data/                        # CIFAR-10 (auto-downloaded)
├── docs/                        # Project documentation
│   └── PROJECT_REFERENCE.md     # ← this file
├── requirements.txt
└── README.md
```

---

## 4. Configuration Files (`configs/`)

Each YAML file fully defines one experiment. Loading a config is the only argument needed by `scripts/train.py`.

### `BaseFP32.yaml`
Full-precision baseline. No AMP, no regularization beyond weight decay. Provides the reference accuracy ceiling for FP32 with minimal augmentation.

### `BaseFP16.yaml`
Identical to BaseFP32 except `use_amp: true`. Isolates the accuracy and speed effect of enabling PyTorch AMP without changing any other variable.

### `AugmFP32.yaml`
FP32 training with the extended augmentation pipeline and full regularization (label smoothing, gradient clipping, warmup). Establishes the best accuracy achievable in FP32.

### `AugmFP16.yaml`
FP16 + extended augmentation + full regularization. The target "production" configuration — expected to be the fastest while also achieving the highest test accuracy.

### `FP8Test.yaml`
Not a training configuration. Points to the checkpoint produced by AugmFP16 and passes it to the FP8 quantization evaluation script. Accuracy is measured before and after conversion to FP8 E4M3 format.

---

## 5. Source Modules (`src/`)

### `src/models/vit_model.py`

**What it does:** Wraps the `timm` library to create ViT models adapted for CIFAR-10.

| Function | Description |
|----------|-------------|
| `create_vit_model(model_name, num_classes, pretrained)` | Instantiates a named ViT variant. All experiments use `vit_tiny_patch16_224`. |
| `count_parameters(model)` | Returns total and trainable parameter counts. |
| `get_model_info(model)` | Returns a dict with parameter count in millions and layer summary. |

Supported variants: `vit_tiny` (5.7 M), `vit_small` (22 M), `vit_base` (86 M).

---

### `src/data/dataset.py`

**What it does:** Downloads and loads CIFAR-10; applies the augmentation pipeline specified in the config.

| Function | Description |
|----------|-------------|
| `get_cifar10_loaders(batch_size, num_workers, augmentation, data_dir, pin_memory)` | Returns `(train_loader, test_loader)`. `augmentation` is `"basic"` or `"extended"`. |
| `get_dataset_info()` | Returns metadata dict: class names, split sizes, normalization stats. |

**Image pipeline:**
- All images are resized from 32×32 → 224×224 (ViT requires 224×224 input).
- Normalization uses CIFAR-10 channel statistics: mean = `[0.4914, 0.4822, 0.4465]`, std = `[0.2470, 0.2435, 0.2616]`.

---

### `src/training/trainer.py`

**What it does:** Implements `ViTTrainer`, the central training loop, handling AMP, monitoring, checkpointing, and metric tracking.

| Method | Description |
|--------|-------------|
| `train_epoch()` | Runs one forward + backward pass over the training set. Returns `(avg_loss, avg_accuracy)`. |
| `validate()` | Evaluates on the test set with no gradients. Returns `(avg_loss, avg_accuracy)`. |
| `train(num_epochs, save_every)` | Full training loop. Calls `train_epoch` and `validate` each epoch, saves checkpoints, and handles `KeyboardInterrupt` gracefully. |
| `save_checkpoint(epoch, val_acc, is_best)` | Saves model weights, optimizer state, scheduler state, and current metrics. Best model is always written to `best_model.pt`. |

**AMP logic:**
- `use_amp: true` → `torch.amp.autocast()` with `GradScaler` on CUDA/MPS devices, BF16 on CPU.
- `use_amp: false` → Standard FP32 forward/backward with no casting.

**Background monitoring threads** (started automatically):
- `SystemMonitor` (CPU, memory, thermal) — samples every 2 seconds.
- `GPUMonitor` (Apple Silicon GPU via `powermetrics`) — requires `sudo`.

---

### `src/utils/metrics.py`

**What it does:** Accumulates per-epoch metrics and persists them to disk.

| Class / Function | Description |
|------------------|-------------|
| `MetricsTracker.update(**kwargs)` | Appends new values to tracked lists for any named metric. |
| `MetricsTracker.save(filename)` | Writes all accumulated metrics to a JSON file. |
| `MetricsTracker.load(filename)` | Loads metrics from an existing JSON file for post-hoc analysis. |
| `MetricsTracker.get_best_acc()` | Returns the maximum validation accuracy seen so far. |
| `calculate_accuracy(outputs, targets)` | Computes top-1 accuracy from raw logits and integer labels. |
| `Timer` | Context manager; measures and returns wall-clock duration of a code block. |

---

### `src/utils/system_monitor.py`

**What it does:** Background-thread monitor for CPU and memory usage and macOS thermal pressure.

| Method | Description |
|--------|-------------|
| `start()` | Launches the sampling thread. |
| `stop()` | Joins the thread and returns `(summary_dict, full_stats_list)`. |

**Sampled every 2 seconds:**
- `cpu_percent` — CPU utilization across all cores (%)
- `memory_percent` — RAM used (%)
- `thermal_pressure` — macOS thermal state (nominal / fair / serious / critical)
- `timestamp` — Seconds elapsed since monitor start

---

### `src/utils/gpu_monitor.py`

**What it does:** Background-thread monitor for Apple Silicon GPU via macOS `powermetrics`. Requires `sudo`.

| Method | Description |
|--------|-------------|
| `start()` | Prompts for sudo and begins sampling. |
| `stop()` | Joins the thread, writes a CSV, returns collected data. |

**Captured fields (written to `gpu_stats.csv`):**

| Column | Unit | Description |
|--------|------|-------------|
| `timestamp` | HH:MM:SS | Wall clock time of sample |
| `gpu_utilization_percent` | % | Fraction of GPU active (residency) |
| `gpu_power_mW` | mW | GPU package power consumption |
| `cpu_power_mW` | mW | CPU package power consumption |

---

## 6. Scripts (`scripts/`)

### `scripts/train.py`

**What it does:** Entry point for a complete training run.

**Workflow:**
1. Parse `--config` argument, load YAML.
2. Create ViT model via `create_vit_model`.
3. Build data loaders via `get_cifar10_loaders`.
4. Instantiate `ViTTrainer`.
5. Call `trainer.train(num_epochs)`.
6. Save `final_metrics.json`, `timing_report.json`, `hardware_stats.json`.

**Usage:**
```bash
python scripts/train.py --config configs/AugmFP16.yaml
python scripts/train.py --config configs/BaseFP32.yaml --device cpu
```

---

### `scripts/evaluate_fp8_quantization.py`

**What it does:** Measures accuracy degradation from post-training quantization to FP8.

**Workflow:**
1. Load the best checkpoint from AugmFP16 (or path specified in `FP8Test.yaml`).
2. Evaluate on the CIFAR-10 test set → **original accuracy**.
3. Convert all linear-layer weights to FP8 E4M3 format (simulate quantization).
4. Cast weights back to FP16 for inference.
5. Evaluate again → **quantized accuracy**.
6. Compute and save degradation metrics to `results/FP8Test/metrics/fp8_quantization_results.json`.

---

### `scripts/extract_metrics.py`

**What it does:** Reads completed experiment directories and produces summary statistics and comparison tables.

**Key outputs:**
- Per-experiment accuracy/timing summary printed to console.
- `comparison_table.csv` — tab-separated table of all experiments side by side.
- `experiment_comparison.json` — machine-readable structured comparison used by plotting scripts.

**Usage:**
```bash
python scripts/extract_metrics.py --all
python scripts/extract_metrics.py --experiment results/BaseFP32
python scripts/extract_metrics.py --compare results/BaseFP32 results/AugmFP16
```

---

### `scripts/compare_experiments.py`

**What it does:** Loads `experiment_comparison.json` and renders detailed cross-experiment analysis tables to the console, covering performance, timing, hardware, and convergence.

---

### `scripts/generate_plots.py`

**What it does:** Produces publication-quality PNG figures (300 DPI) for a single experiment or all experiments.

**Plots generated per experiment:**
- Training loss curve + validation loss curve (overlaid)
- Training accuracy curve + validation accuracy curve (overlaid)
- Learning rate schedule
- CPU, memory, and thermal pressure over training time

**Cross-experiment comparison plots** (saved to `results/comparison_plots/`):
- Best validation accuracy bar chart
- Average epoch time bar chart
- Overfitting gap comparison
- Convergence speed comparison

---

### `scripts/plot_hardware_stats.py`

**What it does:** Standalone visualization of hardware monitoring data. Reads `hardware_stats.json` and `gpu_stats.csv` for one or more experiments and plots time-series graphs of CPU, memory, thermal, GPU utilization, and GPU power.

---

## 7. Results Layout (`results/`)

Each experiment produces the following structure:

```
results/{ExperimentName}/
├── checkpoints/
│   ├── best_model.pt               # Weights at best validation accuracy
│   └── checkpoint_epoch_N.pt       # Periodic snapshots
└── metrics/
    ├── final_metrics.json          # Per-epoch training/validation metrics
    ├── timing_report.json          # Epoch timing statistics
    ├── hardware_stats.json         # CPU / memory / thermal summary + time series
    └── gpu_stats.csv               # GPU utilization and power (if sudo granted)
```

Shared outputs (written after all experiments complete):
```
results/
├── experiment_comparison.json      # Structured summary of all experiments
├── comparison_table.csv            # Human-readable cross-experiment table
└── comparison_plots/               # Multi-experiment visualization PNGs
```

---

## 8. Metrics Reference

### Per-Epoch Training Metrics

Stored as parallel lists in `final_metrics.json`. Each list has one entry per epoch.

| Key | Type | Description |
|-----|------|-------------|
| `train_loss` | `float[]` | Average cross-entropy loss on the training set |
| `train_acc` | `float[]` | Top-1 accuracy on the training set (%) |
| `val_loss` | `float[]` | Average cross-entropy loss on the test set |
| `val_acc` | `float[]` | Top-1 accuracy on the test set (%) |
| `epoch_time` | `float[]` | Wall-clock duration of the epoch (seconds) |
| `learning_rates` | `float[]` | Learning rate used at the start of each epoch |

---

### Derived Summary Metrics

Computed from the per-epoch lists by `extract_metrics.py` and stored in `experiment_comparison.json`.

| Metric | Formula / Source | Interpretation |
|--------|-----------------|----------------|
| `best_val_acc` | `max(val_acc)` | Peak test accuracy — primary performance indicator |
| `best_val_acc_epoch` | `argmax(val_acc)` | Epoch at which peak accuracy was reached |
| `final_val_acc` | `val_acc[-1]` | Test accuracy at the last epoch |
| `final_train_acc` | `train_acc[-1]` | Train accuracy at the last epoch |
| `final_val_loss` | `val_loss[-1]` | Validation loss at the last epoch |
| `final_train_loss` | `train_loss[-1]` | Training loss at the last epoch |
| `overfitting_gap` | `final_train_acc − final_val_acc` | Generalization gap (lower is better) |
| `overfitting_score` | `best_val_acc − final_val_acc` | Late-epoch accuracy regression (lower is better) |
| `convergence_epoch` | First epoch where `val_acc ≥ best_val_acc − 0.5` | How quickly the model reaches near-optimal accuracy |

---

### Timing Metrics

Derived from `epoch_time` list; stored in `timing_report.json` and `experiment_comparison.json`.

| Metric | Unit | Description |
|--------|------|-------------|
| `total_time_hours` | hours | Total wall-clock training time |
| `total_time_minutes` | minutes | Same, in minutes |
| `avg_epoch_time_sec` | seconds | Mean duration per epoch |
| `std_epoch_time_sec` | seconds | Std dev of epoch durations (training stability) |
| `min_epoch_time_sec` | seconds | Fastest epoch |
| `max_epoch_time_sec` | seconds | Slowest epoch |

---

### Hardware Utilization Metrics

Sampled continuously by `SystemMonitor` and `GPUMonitor`; saved to `hardware_stats.json` and `gpu_stats.csv`.

| Metric | Unit | Source | Description |
|--------|------|--------|-------------|
| `cpu_avg` | % | `hardware_stats.json` | Mean CPU utilization over entire training run |
| `cpu_max` | % | `hardware_stats.json` | Peak CPU utilization |
| `mem_avg` | % | `hardware_stats.json` | Mean RAM usage |
| `mem_max` | % | `hardware_stats.json` | Peak RAM usage |
| `thermal_max` | level | `hardware_stats.json` | Highest macOS thermal pressure level observed |
| `thermal_throttled` | bool | `hardware_stats.json` | Whether the CPU was thermally throttled at any point |
| `gpu_utilization_percent` | % | `gpu_stats.csv` | GPU active residency (Apple Silicon) |
| `gpu_power_mW` | mW | `gpu_stats.csv` | GPU power draw |
| `cpu_power_mW` | mW | `gpu_stats.csv` | CPU power draw (from `powermetrics`) |

---

### FP8 Quantization Metrics

Produced by `evaluate_fp8_quantization.py`; saved to `results/FP8Test/metrics/fp8_quantization_results.json`.

| Metric | Description |
|--------|-------------|
| `original_accuracy` | Test accuracy of the FP16 model before quantization (%) |
| `quantized_accuracy` | Test accuracy after converting weights to FP8 E4M3 and back (%) |
| `accuracy_degradation` | `original_accuracy − quantized_accuracy` (percentage points) |
| `relative_degradation` | `accuracy_degradation / original_accuracy` (fraction) |

---

## 9. What Is Being Compared

The experiments answer the following specific questions. Each comparison isolates a single variable.

### Precision effect (basic augmentation only)
**BaseFP32 vs BaseFP16**

| Compared metric | What it reveals |
|-----------------|----------------|
| `best_val_acc` | Accuracy cost of FP16 |
| `avg_epoch_time_sec` | Training speedup from AMP |
| `overfitting_gap` | Whether precision affects overfitting |

### Augmentation effect (FP32 only)
**BaseFP32 vs AugmFP32**

| Compared metric | What it reveals |
|-----------------|----------------|
| `best_val_acc` | Accuracy gain from extended augmentation |
| `overfitting_gap` | Regularization effectiveness |
| `convergence_epoch` | Whether augmentation slows convergence |

### Augmentation effect (FP16 only)
**BaseFP16 vs AugmFP16**

Same metrics as above, measured under FP16.

### Interaction: precision × augmentation
**BaseFP32 vs AugmFP16** (opposite corners of the factorial)

| Compared metric | What it reveals |
|-----------------|----------------|
| `best_val_acc` | Net gain from combining both improvements |
| `total_time_hours` | End-to-end time saving |
| `cpu_avg`, `gpu_utilization_percent` | Hardware efficiency |

### FP8 quantization
**AugmFP16 (baseline) vs FP8Test**

| Compared metric | What it reveals |
|-----------------|----------------|
| `original_accuracy` vs `quantized_accuracy` | Accuracy cost of FP8 at inference |
| `accuracy_degradation` | Whether FP8 is viable for deployment |
