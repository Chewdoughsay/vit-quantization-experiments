"""
Phase 1 — FP16 static quantization evaluation on ImageNet-1k validation.

Loads ViT-Tiny pretrained on ImageNet-1k, evaluates FP32 baseline vs FP16
(``model.half()``).  Downloads ImageNet-1k validation split (~6 GB) from
HuggingFace on first run (requires ``huggingface-cli login``).

Outputs:
    results/FP16ImageNet/metrics/fp16_imagenet_results.json

Usage:
    python scripts/evaluate_fp16_imagenet.py
    python scripts/evaluate_fp16_imagenet.py --device mps --batch-size 64
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


def main():
    parser = argparse.ArgumentParser(description="FP16 static evaluation on ImageNet-1k validation")
    parser.add_argument("--device",      type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("FP16 Static Quantization  —  ViT-Tiny on ImageNet-1k validation")
    print(f"PyTorch {torch.__version__}  |  device: {args.device}")
    print("=" * 70 + "\n")

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # STEP 1: FP32 baseline — load model first to get correct preprocessing
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: FP32 baseline  (pretrained=True, model in float32)")
    print("=" * 70)

    model_fp32 = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    model_fp32 = model_fp32.to(device).eval()

    # Use timm's own data config — avoids hardcoding wrong mean/std
    data_config = timm.data.resolve_data_config({}, model=model_fp32)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    print(f"  timm data config: {data_config}\n")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    loader = load_imagenet_val(val_transform, args.batch_size, args.num_workers)

    n_val = len(loader.dataset)
    print(f"Validation: {n_val} images  |  {len(loader)} batches  |  bs={args.batch_size}\n")
    fp32_mem = model_size_mb(model_fp32)
    print(f"Model size (FP32): {fp32_mem:.2f} MB\n")

    fp32_results = evaluate(model_fp32, loader, device, desc="FP32")
    print(f"\n  Accuracy : {fp32_results['accuracy_percent']:.2f}%")
    print(f"  Loss     : {fp32_results['avg_loss']:.4f}")
    print(f"  Latency  : {fp32_results['avg_latency_ms_per_batch']:.2f} ms/batch\n")

    del model_fp32

    # ------------------------------------------------------------------
    # STEP 2: FP16  (model.half())
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 2: FP16 static  (model.half()  —  all weights cast to float16)")
    print("=" * 70)

    model_fp16 = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)
    model_fp16 = model_fp16.half().to(device).eval()
    fp16_mem = model_size_mb(model_fp16)
    print(f"Model size (FP16): {fp16_mem:.2f} MB  (reduction: {fp32_mem / fp16_mem:.2f}x)\n")

    fp16_error = None
    try:
        fp16_results = evaluate(model_fp16, loader, device, desc="FP16")
    except RuntimeError as exc:
        fp16_error = str(exc)
        print(f"\nWARNING: FP16 on {args.device} failed ({exc})")
        print("Falling back to CPU for FP16 evaluation ...\n")
        device_fp16 = torch.device("cpu")
        model_fp16  = model_fp16.to(device_fp16)
        fp16_results = evaluate(model_fp16, loader, device_fp16, desc="FP16 (CPU fallback)")

    print(f"\n  Accuracy : {fp16_results['accuracy_percent']:.2f}%")
    print(f"  Loss     : {fp16_results['avg_loss']:.4f}")
    print(f"  Latency  : {fp16_results['avg_latency_ms_per_batch']:.2f} ms/batch\n")

    del model_fp16

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    degradation_pp = fp32_results["accuracy_percent"] - fp16_results["accuracy_percent"]
    lat_fp32 = fp32_results["avg_latency_ms_per_batch"]
    lat_fp16 = fp16_results["avg_latency_ms_per_batch"]
    speedup  = (lat_fp32 / lat_fp16) if lat_fp16 > 0 else None

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "experiment": "FP16ImageNet",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "model": "vit_tiny_patch16_224",
        "pretrained": True,
        "dataset": "ImageNet-1k validation (50000 images, 1000 classes)",
        "device": args.device,
        "batch_size": args.batch_size,
        "fp32": {
            **fp32_results,
            "memory_mb": round(fp32_mem, 3),
        },
        "fp16": {
            **fp16_results,
            "memory_mb": round(fp16_mem, 3),
            "conversion": "model.half()",
            "device_fallback_error": fp16_error,
        },
        "comparison": {
            "accuracy_degradation_pp": round(degradation_pp, 4),
            "memory_reduction_ratio": round(fp32_mem / fp16_mem, 3),
            "latency_speedup": round(speedup, 3) if speedup else None,
        },
    }

    output_dir = Path("results/FP16ImageNet/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "fp16_imagenet_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"\n{'Method':<10} {'Accuracy':>10} {'Loss':>10} {'Latency (ms)':>14} {'Memory (MB)':>12}"
    print(header)
    print("-" * 60)
    print(f"{'FP32':<10} {fp32_results['accuracy_percent']:>9.2f}%  "
          f"{fp32_results['avg_loss']:>10.4f}  "
          f"{lat_fp32:>13.2f}  {fp32_mem:>11.2f}")
    print(f"{'FP16':<10} {fp16_results['accuracy_percent']:>9.2f}%  "
          f"{fp16_results['avg_loss']:>10.4f}  "
          f"{lat_fp16:>13.2f}  {fp16_mem:>11.2f}")
    print(f"\nAccuracy degradation : {degradation_pp:+.4f} pp")
    print(f"Memory reduction     : {fp32_mem / fp16_mem:.2f}x")
    if speedup:
        print(f"Latency speedup      : {speedup:.2f}x")
    if fp16_error:
        print(f"\nNote: FP16 ran on CPU fallback  ({args.device} does not support all float16 ops)")
    print(f"\nResults saved → {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
