"""
Phase 2 — INT8 weight-only quantization evaluation on ImageNet-1k validation.

Selective quantization: only nn.Linear layers in Attention (qkv, proj) and
MLP (fc1, fc2) are quantized with per-tensor linear scaling.  LayerNorm,
cls_token, pos_embed, and the classification head are kept in float32.

Compares FP32 baseline vs INT8: accuracy, latency, memory, and per-layer
quantization error (MSE, MAE, scale).

Outputs:
    results/INT8ImageNet/metrics/int8_imagenet_results.json

Usage:
    python scripts/evaluate_int8_quantization.py
    python scripts/evaluate_int8_quantization.py --device mps --batch-size 64
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import timm.data
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.imagenet_loader import load_imagenet_val
from src.evaluation.evaluator import evaluate, model_size_mb
from src.models.vit_model import create_vit_model
from src.models.quantized_linear import quantize_model_selective, SKIP_PATTERNS


def main():
    parser = argparse.ArgumentParser(description="INT8 static quantization on ImageNet-1k validation")
    parser.add_argument("--device",      type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("INT8 Static Quantization (per-tensor)  —  ViT-Tiny on ImageNet-1k")
    print(f"PyTorch {torch.__version__}  |  device: {args.device}")
    print("=" * 70 + "\n")

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    _tmp_model = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=False)
    data_config   = timm.data.resolve_data_config({}, model=_tmp_model)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    del _tmp_model

    loader = load_imagenet_val(val_transform, args.batch_size, args.num_workers)
    print(f"Validation: {len(loader.dataset)} images  |  {len(loader)} batches  |  bs={args.batch_size}\n")

    # ------------------------------------------------------------------
    # STEP 1: FP32 baseline
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: FP32 baseline")
    print("=" * 70)

    model_fp32 = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    model_fp32 = model_fp32.to(device).eval()
    fp32_mem   = model_size_mb(model_fp32)
    print(f"Model size (FP32): {fp32_mem:.2f} MB\n")

    fp32_results = evaluate(model_fp32, loader, device, desc="FP32")
    print(f"\n  Accuracy : {fp32_results['accuracy_percent']:.2f}%")
    print(f"  Loss     : {fp32_results['avg_loss']:.4f}")
    print(f"  Latency  : {fp32_results['avg_latency_ms_per_batch']:.2f} ms/batch\n")

    del model_fp32

    # ------------------------------------------------------------------
    # STEP 2: INT8 cuantizare selectivă
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 2: INT8 static (weight-only, per-tensor, linear scaling)")
    print(f"  Skip patterns: {SKIP_PATTERNS}")
    print("=" * 70)

    model_int8 = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)

    # Quantize on CPU (numpy-like ops), then move to device
    model_int8, layer_stats = quantize_model_selective(model_int8, verbose=True)
    model_int8 = model_int8.to(device).eval()

    int8_mem = model_size_mb(model_int8)
    print(f"\nModel size (INT8): {int8_mem:.2f} MB  (reduction: {fp32_mem / int8_mem:.2f}x)\n")

    int8_results = evaluate(model_int8, loader, device, desc="INT8")
    print(f"\n  Accuracy : {int8_results['accuracy_percent']:.2f}%")
    print(f"  Loss     : {int8_results['avg_loss']:.4f}")
    print(f"  Latency  : {int8_results['avg_latency_ms_per_batch']:.2f} ms/batch\n")

    del model_int8

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    degradation = fp32_results["accuracy_percent"] - int8_results["accuracy_percent"]
    lat_fp32    = fp32_results["avg_latency_ms_per_batch"]
    lat_int8    = int8_results["avg_latency_ms_per_batch"]
    speedup     = (lat_fp32 / lat_int8) if lat_int8 > 0 else None

    # Per-layer summary stats
    total_q_params = sum(s["n_params"] for s in layer_stats)
    avg_mse        = sum(s["mse"] for s in layer_stats) / len(layer_stats)
    worst_layer    = max(layer_stats, key=lambda s: s["mse"])
    best_layer     = min(layer_stats, key=lambda s: s["mse"])

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "experiment":      "INT8ImageNet",
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "model":           "vit_tiny_patch16_224",
        "pretrained":      True,
        "dataset":         "ImageNet-1k validation (50000 images, 1000 classes)",
        "device":          args.device,
        "batch_size":      args.batch_size,
        "quantization": {
            "method":          "per-tensor INT8 (linear scaling)",
            "skip_patterns":   SKIP_PATTERNS,
            "n_layers_quant":  len(layer_stats),
            "total_q_params":  total_q_params,
            "avg_mse":         round(avg_mse, 8),
            "worst_layer":     worst_layer["layer"],
            "worst_layer_mse": round(worst_layer["mse"], 8),
            "best_layer":      best_layer["layer"],
            "best_layer_mse":  round(best_layer["mse"], 8),
        },
        "fp32": {
            **fp32_results,
            "memory_mb": round(fp32_mem, 3),
        },
        "int8": {
            **int8_results,
            "memory_mb": round(int8_mem, 3),
        },
        "comparison": {
            "accuracy_degradation_pp": round(degradation, 4),
            "memory_reduction_ratio":  round(fp32_mem / int8_mem, 3),
            "latency_speedup":         round(speedup, 3) if speedup else None,
        },
        "layer_stats": layer_stats,
    }

    output_dir  = Path("results/INT8ImageNet/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "int8_imagenet_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<10} {'Accuracy':>10} {'Loss':>10} {'Latency (ms)':>14} {'Memory (MB)':>12}")
    print("-" * 60)
    print(f"{'FP32':<10} {fp32_results['accuracy_percent']:>9.2f}%  "
          f"{fp32_results['avg_loss']:>10.4f}  "
          f"{lat_fp32:>13.2f}  {fp32_mem:>11.2f}")
    print(f"{'INT8':<10} {int8_results['accuracy_percent']:>9.2f}%  "
          f"{int8_results['avg_loss']:>10.4f}  "
          f"{lat_int8:>13.2f}  {int8_mem:>11.2f}")
    print(f"\nAccuracy degradation : {degradation:+.4f} pp")
    print(f"Memory reduction     : {fp32_mem / int8_mem:.2f}x")
    if speedup:
        print(f"Latency speedup      : {speedup:.2f}x")
    print(f"\nQuantized layers     : {len(layer_stats)}")
    print(f"Avg MSE per layer    : {avg_mse:.4e}")
    print(f"Worst layer (MSE)    : {worst_layer['layer']}  ({worst_layer['mse']:.4e})")
    print(f"\nResults saved → {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
