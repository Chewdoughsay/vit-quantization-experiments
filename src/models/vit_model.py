"""
Vision Transformer model creation via the `timm` library.

This module wraps timm.create_model to provide a single entry point for
loading ViT variants (Tiny / Small / Base) with optional ImageNet-1k
pretrained weights.  It is used throughout the project by both the
preliminary CIFAR-10 training scripts and the Faze 1-3 quantization
evaluation scripts.

Usage:
    from src.models.vit_model import create_vit_model

    # Pretrained on ImageNet-1k, full 1000-class head
    model = create_vit_model("vit_tiny_patch16_224", num_classes=1000, pretrained=True)

    # Random init, 10 classes (CIFAR-10)
    model = create_vit_model(num_classes=10)
"""

import timm
import torch


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_vit_model(
    model_name: str = "vit_tiny_patch16_224",
    num_classes: int = 10,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Create a Vision Transformer model from the timm registry.

    Args:
        model_name:  Any timm model name (default: ViT-Tiny with 16×16 patches).
        num_classes: Number of output classes.  Use 1000 for pretrained ImageNet
                     evaluation, 10 for CIFAR-10.
        pretrained:  If True, load ImageNet-1k weights from timm.

    Returns:
        A ready-to-use nn.Module.
    """
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_info(model: torch.nn.Module) -> dict:
    """Return a dict with parameter counts in absolute and millions."""
    total, trainable = count_parameters(model)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "total_params_millions": total / 1e6,
        "trainable_params_millions": trainable / 1e6,
    }


# ---------------------------------------------------------------------------
# Quick-reference catalogue of ViT sizes used in this project
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "vit_tiny": {
        "name": "vit_tiny_patch16_224",
        "params_approx": "5.7M",
        "description": "ViT-Tiny — used for all Faze 1-3 experiments",
    },
    "vit_small": {
        "name": "vit_small_patch16_224",
        "params_approx": "22M",
        "description": "ViT-Small — planned for Faza 4 (scaling study)",
    },
    "vit_base": {
        "name": "vit_base_patch16_224",
        "params_approx": "86M",
        "description": "ViT-Base — planned for Faza 4 (scaling study)",
    },
}


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}  |  MPS: {torch.backends.mps.is_available()}\n")

    for key, cfg in MODEL_CONFIGS.items():
        model = create_vit_model(cfg["name"], num_classes=1000, pretrained=False)
        info = get_model_info(model)
        out = model(torch.randn(1, 3, 224, 224))
        print(f"{key:<12}  {info['trainable_params_millions']:.1f}M params  "
              f"output={out.shape}  ({cfg['description']})")
