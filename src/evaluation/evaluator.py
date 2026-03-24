"""
Shared evaluation loop and model-size helpers — used by all phases.

Handles FP16 dtype detection, device synchronization (MPS / CUDA),
warmup batches, and consistent metric keys across scripts.
"""

import os
import tempfile
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Eval",
    warmup_batches: int = 3,
    leave_tqdm: bool = True,
) -> dict:
    """Evaluate accuracy, loss, and latency.

    Automatically detects FP16 models and casts inputs accordingly.

    Returns:
        Dict with keys ``accuracy_percent``, ``avg_loss``,
        ``avg_latency_ms_per_batch``, ``total_samples``.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    is_half = next(model.parameters()).dtype == torch.float16

    correct, total, total_loss = 0, 0, 0.0
    batch_times: list[float] = []

    for i, (images, labels) in enumerate(tqdm(loader, desc=desc, leave=leave_tqdm)):
        images = images.to(device)
        labels = labels.to(device)
        if is_half:
            images = images.half()

        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= warmup_batches:
            batch_times.append(t1 - t0)

        total_loss += criterion(outputs.float(), labels).item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    avg_lat = (sum(batch_times) / len(batch_times) * 1000) if batch_times else 0.0

    return {
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 4),
        "avg_loss": round(total_loss / len(loader), 6),
        "avg_latency_ms_per_batch": round(avg_lat, 3),
        "total_samples": total,
    }


def model_size_mb(model: nn.Module) -> float:
    """Parameter + buffer memory in MB (based on dtype element size)."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def model_disk_mb(model: nn.Module) -> float:
    """Serialized state_dict size in MB (via ``torch.save`` to a temp file)."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 ** 2)
    os.unlink(path)
    return round(size, 3)
