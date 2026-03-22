"""
FP8 E4M3FN post-training quantization evaluation (per-tensor and per-channel).

**Legacy** — part of the preliminary CIFAR-10 study.  Compares FP8 E4M3FN
quantization at two granularities against the AugmFP16 checkpoint baseline.
Requires the AugmFP16 experiment to have been trained first.

Usage:
    python scripts/preliminary/evaluate_fp8_quantization.py
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

# Add project root so ``src.*`` imports work when running from any directory.
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_model import create_vit_model
from src.data.dataset import get_cifar10_loaders
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# FP8 quantization primitives
# ═══════════════════════════════════════════════════════════════════════════

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0 for E4M3FN


def quantize_weight_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 E4M3FN using a single global scale factor."""
    abs_max = tensor.abs().max().clamp(min=1e-12)
    scale = FP8_MAX / abs_max
    # FP8 cast must happen on CPU (MPS does not support float8_e4m3fn)
    cpu = tensor.cpu()
    quantized = (cpu * scale.cpu()).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized.to(torch.float8_e4m3fn).to(cpu.dtype)
    return (quantized / scale.cpu()).to(tensor.device)


def quantize_weight_per_channel(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 E4M3FN with one scale per output channel (dim 0); 1-D tensors fall back to per-tensor."""
    if tensor.ndim == 1:
        return quantize_weight_per_tensor(tensor)

    reduce_dims = list(range(1, tensor.ndim))
    channel_maxes = tensor.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-12)
    scale = FP8_MAX / channel_maxes

    # FP8 cast must happen on CPU (MPS does not support float8_e4m3fn)
    cpu = tensor.cpu()
    scale_cpu = scale.cpu()
    quantized = (cpu * scale_cpu).clamp(-FP8_MAX, FP8_MAX)
    quantized = quantized.to(torch.float8_e4m3fn).to(cpu.dtype)
    return (quantized / scale_cpu).to(tensor.device)


# ═══════════════════════════════════════════════════════════════════════════
# Model-level quantization
# ═══════════════════════════════════════════════════════════════════════════

def quantize_model(model: torch.nn.Module, granularity: str) -> torch.nn.Module:
    """Quantize all trainable parameters in-place with the given granularity."""
    fn = quantize_weight_per_tensor if granularity == "per_tensor" else quantize_weight_per_channel

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = fn(param.data)

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Per-layer error analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_layer_errors(
    original_params: dict[str, torch.Tensor],
    quantized_model: torch.nn.Module,
) -> dict[str, dict]:
    """Return per-layer MSE, max_abs_error, relative_error, and shape between original and quantized weights."""
    layer_errors = {}
    for name, param in quantized_model.named_parameters():
        if name not in original_params:
            continue
        orig = original_params[name]
        quant = param.data

        mse = F.mse_loss(quant, orig).item()
        max_abs = (quant - orig).abs().max().item()
        orig_power = orig.pow(2).mean().item()
        relative = mse / max(orig_power, 1e-12)

        layer_errors[name] = {
            "mse": mse,
            "max_abs_error": max_abs,
            "relative_error": relative,
            "shape": list(orig.shape),
        }

    return layer_errors


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    desc: str = "Evaluation",
) -> tuple[float, float]:
    """Return (accuracy, avg_loss) on test_loader."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(test_loader, desc=desc):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        total_loss += criterion(outputs, labels).item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_correct / total_samples, total_loss / len(test_loader)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("FP8 Post-Training Quantization Evaluation")
    print(f"Using native torch.float8_e4m3fn  (max = {FP8_MAX})")
    print("=" * 70 + "\n")

    # Configuration
    CONFIG = {
        "name": "FP8Test",
        "source_model": "results/preliminary/AugmFP16/checkpoints/best_model.pt",
        "model_name": "vit_tiny_patch16_224",
        "num_classes": 10,
        "batch_size": 128,
        "device": "mps",
    }

    print("Configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print()

    source_path = Path(CONFIG["source_model"])
    if not source_path.exists():
        print(f"Error: checkpoint not found at {source_path}")
        print("Please run the AugmFP16 experiment first:")
        print("  $ python scripts/train.py --config configs/AugmFP16.yaml")
        sys.exit(1)

    device = torch.device(CONFIG["device"])

    model = create_vit_model(
        model_name=CONFIG["model_name"],
        num_classes=CONFIG["num_classes"],
        pretrained=False,
    )

    checkpoint = torch.load(source_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint  (saved val_acc: {checkpoint['val_acc']:.4f})\n")

    _, test_loader = get_cifar10_loaders(
        batch_size=CONFIG["batch_size"],
        num_workers=2,
        augmentation="extended",
        data_dir="./data",
    )
    print(f"Test loader: {len(test_loader)} batches\n")

    original_params = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    print("=" * 70)
    print("STEP 1: Evaluate original FP16 model")
    print("=" * 70)
    original_acc, original_loss = evaluate_model(model, test_loader, device, desc="FP16 baseline")
    print(f"\n  Accuracy: {original_acc * 100:.2f}%   Loss: {original_loss:.4f}\n")

    print("=" * 70)
    print("STEP 2: Per-tensor FP8 E4M3FN quantization")
    print("=" * 70)
    model = quantize_model(model, granularity="per_tensor")
    pt_acc, pt_loss = evaluate_model(model, test_loader, device, desc="Per-tensor FP8")
    pt_errors = compute_layer_errors(original_params, model)
    pt_degradation = (original_acc - pt_acc) * 100
    print(f"\n  Accuracy: {pt_acc * 100:.2f}%   Loss: {pt_loss:.4f}")
    print(f"  Degradation vs FP16: {pt_degradation:+.2f} pp\n")

    print("=" * 70)
    print("STEP 3: Per-channel FP8 E4M3FN quantization")
    print("=" * 70)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

    model = quantize_model(model, granularity="per_channel")
    pc_acc, pc_loss = evaluate_model(model, test_loader, device, desc="Per-channel FP8")
    pc_errors = compute_layer_errors(original_params, model)
    pc_degradation = (original_acc - pc_acc) * 100
    print(f"\n  Accuracy: {pc_acc * 100:.2f}%   Loss: {pc_loss:.4f}")
    print(f"  Degradation vs FP16: {pc_degradation:+.2f} pp\n")

    results = {
        "experiment": "FP8Test",
        "source_model": str(source_path),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "fp8_dtype": "float8_e4m3fn",
        "fp8_max": FP8_MAX,
        "original_fp16": {
            "accuracy": float(original_acc),
            "accuracy_percent": round(original_acc * 100, 4),
            "loss": float(original_loss),
        },
        "per_tensor": {
            "accuracy": float(pt_acc),
            "accuracy_percent": round(pt_acc * 100, 4),
            "loss": float(pt_loss),
            "degradation_pp": round(float(pt_degradation), 4),
            "relative_degradation": round(float(pt_degradation / (original_acc * 100)), 6),
            "loss_increase": round(float(pt_loss - original_loss), 4),
            "layer_errors": pt_errors,
        },
        "per_channel": {
            "accuracy": float(pc_acc),
            "accuracy_percent": round(pc_acc * 100, 4),
            "loss": float(pc_loss),
            "degradation_pp": round(float(pc_degradation), 4),
            "relative_degradation": round(float(pc_degradation / (original_acc * 100)), 6),
            "loss_increase": round(float(pc_loss - original_loss), 4),
            "layer_errors": pc_errors,
        },
        "quantization_config": {
            "format": "e4m3fn",
            "fp8_max": FP8_MAX,
            "quantize_weights": True,
            "quantize_activations": False,
            "granularities_compared": ["per_tensor", "per_channel"],
        },
    }

    output_dir = Path("results/preliminary/FP8Test/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "fp8_quantization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Accuracy':>10} {'Degradation':>14} {'Loss':>10}")
    print("-" * 58)
    print(f"{'FP16 (original)':<20} {original_acc*100:>9.2f}%  {'—':>14} {original_loss:>10.4f}")
    print(f"{'FP8 per-tensor':<20} {pt_acc*100:>9.2f}%  {pt_degradation:>+13.2f}pp {pt_loss:>10.4f}")
    print(f"{'FP8 per-channel':<20} {pc_acc*100:>9.2f}%  {pc_degradation:>+13.2f}pp {pc_loss:>10.4f}")

    improvement = pt_degradation - pc_degradation
    print(f"\nPer-channel reduces degradation by {improvement:+.2f} pp vs per-tensor")

    def interpret(degradation_pp):
        if degradation_pp < 1.0:
            return "Excellent — FP8 highly viable for deployment"
        elif degradation_pp < 3.0:
            return "Good — FP8 viable for deployment"
        elif degradation_pp < 5.0:
            return "Moderate — consider QAT"
        else:
            return "High degradation — FP8 may not be suitable without QAT"

    print(f"\nPer-tensor:  {interpret(pt_degradation)}")
    print(f"Per-channel: {interpret(pc_degradation)}")
    print(f"\nResults saved to: {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
