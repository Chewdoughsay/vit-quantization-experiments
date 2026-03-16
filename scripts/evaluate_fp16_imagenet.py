"""
FP16 static quantization evaluation on ImageNette (ImageNet proxy).

Loads ViT-Tiny pretrained on ImageNet-1k, evaluates FP32 vs FP16 (model.half()).
Auto-downloads ImageNette2-320 (~330 MB) if not already present.

Usage:
    python scripts/evaluate_fp16_imagenet.py
    python scripts/evaluate_fp16_imagenet.py --data-dir ./data --device mps --batch-size 64
"""

import argparse
import json
import ssl
import sys
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import timm.data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_model import create_vit_model

# ---------------------------------------------------------------------------
# ImageNette config
# ---------------------------------------------------------------------------

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

# WordNet ID -> ImageNet-1k class index (0-based)
# These are the 10 classes in ImageNette and their position in ImageNet-1k
WNID_TO_IMAGENET = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}



# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def download_imagenette(data_dir: Path) -> Path:
    """Download and extract ImageNette2-320 if not already present."""
    dataset_dir = data_dir / "imagenette2-320"
    if dataset_dir.exists():
        print(f"ImageNette already present at: {dataset_dir}")
        return dataset_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "imagenette2-320.tgz"

    print(f"Downloading ImageNette2-320 (~330 MB) ...")

    # macOS (Python from python.org) lacks system SSL certs — bypass verification
    # for this known, trusted URL
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
    with opener.open(IMAGENETTE_URL) as resp, open(archive_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block = 1 << 14  # 16 KB
        while chunk := resp.read(block):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = min(100, downloaded * 100 // total)
                print(f"\r  {pct:3d}%  {downloaded // 1_000_000} / {total // 1_000_000} MB",
                      end="", flush=True)
    print()
    print()

    print("Extracting ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    archive_path.unlink()

    print(f"Ready: {dataset_dir}\n")
    return dataset_dir


def build_label_map(dataset: datasets.ImageFolder) -> list[int]:
    """Return a list where list[local_idx] = ImageNet-1k class index."""
    # class_to_idx: {wnid: local_idx, ...}  (sorted alphabetically by torchvision)
    mapping = [None] * len(dataset.classes)
    for wnid, local_idx in dataset.class_to_idx.items():
        imagenet_idx = WNID_TO_IMAGENET.get(wnid)
        if imagenet_idx is None:
            raise ValueError(f"Unknown WordNet ID in dataset: '{wnid}'")
        mapping[local_idx] = imagenet_idx
    return mapping


def get_val_loader(dataset_dir: Path, batch_size: int, num_workers: int,
                   transform) -> tuple:
    """Return (DataLoader, label_map) for the ImageNette validation split."""
    val_dataset = datasets.ImageFolder(str(dataset_dir / "val"), transform=transform)
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, build_label_map(val_dataset)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    label_map: list[int],
    device: torch.device,
    desc: str = "Eval",
    warmup_batches: int = 3,
) -> dict:
    """Evaluate model accuracy, loss, and latency. Returns a result dict."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    is_half = next(model.parameters()).dtype == torch.float16
    label_map_t = torch.tensor(label_map, dtype=torch.long, device=device)  # fast lookup

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    batch_times = []

    for batch_idx, (images, local_labels) in enumerate(tqdm(loader, desc=desc)):
        images = images.to(device)
        if is_half:
            images = images.half()
        # Remap local class indices (0-9) -> ImageNet-1k indices (0-999)
        imagenet_labels = label_map_t[local_labels.to(device)]

        t0 = time.perf_counter()
        outputs = model(images)
        # Force MPS / CUDA kernel completion before stopping the clock
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if batch_idx >= warmup_batches:
            batch_times.append(t1 - t0)

        loss = criterion(outputs.float(), imagenet_labels)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == imagenet_labels).sum().item()
        total_samples += imagenet_labels.size(0)
        total_loss += loss.item()

    accuracy = total_correct / total_samples
    avg_latency_ms = (sum(batch_times) / len(batch_times) * 1000) if batch_times else 0.0

    return {
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 4),
        "avg_loss": round(total_loss / len(loader), 6),
        "avg_latency_ms_per_batch": round(avg_latency_ms, 3),
        "total_samples": total_samples,
    }


def model_size_mb(model: nn.Module) -> float:
    """Parameter memory in MB (based on dtype element size)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FP16 static evaluation on ImageNette (ImageNet proxy)")
    parser.add_argument("--data-dir",    type=str, default="./data")
    parser.add_argument("--device",      type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("FP16 Static Quantization  —  ViT-Tiny on ImageNette (ImageNet proxy)")
    print(f"PyTorch {torch.__version__}  |  device: {args.device}")
    print("=" * 70 + "\n")

    data_dir = Path(args.data_dir)
    device   = torch.device(args.device)

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
    # Dataset (created after model so we have the correct transform)
    # ------------------------------------------------------------------
    dataset_dir = download_imagenette(data_dir)
    loader, label_map = get_val_loader(dataset_dir, args.batch_size, args.num_workers,
                                       transform=val_transform)

    n_val = len(loader.dataset)
    print(f"Validation: {n_val} images  |  {len(loader)} batches  |  bs={args.batch_size}")
    print(f"Label map (local → ImageNet-1k): {label_map}\n")
    fp32_mem = model_size_mb(model_fp32)
    print(f"Model size (FP32): {fp32_mem:.2f} MB\n")

    fp32_results = evaluate(model_fp32, loader, label_map, device, desc="FP32")
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
        fp16_results = evaluate(model_fp16, loader, label_map, device, desc="FP16")
    except RuntimeError as exc:
        # Some MPS / CUDA ops may not support float16 — fall back to CPU
        fp16_error = str(exc)
        print(f"\nWARNING: FP16 on {args.device} failed ({exc})")
        print("Falling back to CPU for FP16 evaluation ...\n")
        device_fp16 = torch.device("cpu")
        model_fp16  = model_fp16.to(device_fp16)
        fp16_results = evaluate(model_fp16, loader, label_map, device_fp16, desc="FP16 (CPU fallback)")

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
        "dataset": "imagenette2-320 (10-class ImageNet proxy)",
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