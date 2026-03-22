"""
INT8 weight-only quantization with linear scaling.

Provides two granularities for post-training quantization (PTQ) of nn.Linear
layers inside a Vision Transformer:

    Per-tensor:   one scale factor per weight matrix   (simpler, slightly less accurate)
    Per-channel:  one scale factor per output row      (better for layers with outliers)

Quantization formula (per-tensor):

    scale = max(|W|) / 127
    q     = round(W / scale).clamp(-128, 127)   →  stored as torch.int8
    W_hat = q.float() * scale                   →  dequantized at inference

This guarantees a bijective mapping to 256 discrete values, unlike
torch.float8_e4m3fn where not all intermediate values are representable.

Modules:
    QuantizedLinear           — drop-in nn.Linear replacement (per-tensor)
    QuantizedLinearPerChannel — drop-in nn.Linear replacement (per-channel)

Functions:
    int8_quantize / int8_dequantize           — per-tensor primitives
    int8_quantize_per_channel                 — per-channel primitive
    quantize_model_selective                  — quantize all Linear layers except norm/head
    quantize_model_per_channel                — same, but per-channel

Usage:
    from src.models.quantized_linear import quantize_model_selective

    model, stats = quantize_model_selective(model)   # in-place, returns per-layer MSE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Per-tensor quantization primitives
# ═══════════════════════════════════════════════════════════════════════════

def int8_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize a float tensor to int8 with a single scale factor.

    Args:
        tensor: Weight matrix of any shape (typically [out_features, in_features]).

    Returns:
        q:     int8 tensor, same shape.
        scale: float — multiply q by this to approximate the original values.
    """
    max_val = tensor.abs().max().item()
    scale = max_val / 127.0 if max_val > 0 else 1.0
    q = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def int8_dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Recover a float32 approximation from an int8 tensor and its scale."""
    return q.float() * scale


def quantization_error(original: torch.Tensor, q: torch.Tensor, scale: float) -> dict:
    """Measure reconstruction quality: MSE, MAE, and max absolute error."""
    reconstructed = int8_dequantize(q, scale)
    diff = original.float() - reconstructed
    return {
        "mse": diff.pow(2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs_error": diff.abs().max().item(),
        "scale": scale,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-channel quantization primitives
# ═══════════════════════════════════════════════════════════════════════════

def int8_quantize_per_channel(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with one scale factor per output row (per-channel).

    For a weight of shape [out_features, in_features], each row gets its own
    scale factor, which reduces error for layers with non-uniform weight
    distributions (e.g. blocks.7.mlp.*).

    Returns:
        q:      int8 tensor, same shape.
        scales: float32 tensor of shape [out_features, 1].
    """
    max_vals = tensor.float().abs().max(dim=1, keepdim=True).values
    scales = (max_vals / 127.0).clamp(min=1e-8)
    q = (tensor.float() / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales


def quantization_error_per_channel(
    original: torch.Tensor,
    q: torch.Tensor,
    scales: torch.Tensor,
) -> dict:
    """MSE / MAE / max error for per-channel quantization."""
    reconstructed = q.float() * scales
    diff = original.float() - reconstructed
    return {
        "mse": diff.pow(2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_abs_error": diff.abs().max().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# QuantizedLinear — per-tensor wrapper
# ═══════════════════════════════════════════════════════════════════════════

class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with int8 weights (per-tensor scale).

    Weights are stored as int8 buffers and dequantized to float32 on every
    forward pass.  This saves memory (~4x vs float32) at the cost of a small
    dequantization overhead per forward call.

    Typical construction via the factory class method:
        ql = QuantizedLinear.from_linear(existing_linear_layer)
    """

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scale: float,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale

        self.register_buffer("weight_int8", weight_int8)
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().float())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> QuantizedLinear:
        """Build a QuantizedLinear from an existing nn.Linear."""
        q, scale = int8_quantize(linear.weight.data)
        return cls(
            weight_int8=q,
            scale=scale,
            bias=linear.bias.data if linear.bias is not None else None,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp32 = int8_dequantize(self.weight_int8, self.scale)
        return F.linear(x, weight_fp32, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"scale={self.scale:.6f}, dtype=int8")


# ═══════════════════════════════════════════════════════════════════════════
# QuantizedLinearPerChannel — per-channel wrapper
# ═══════════════════════════════════════════════════════════════════════════

class QuantizedLinearPerChannel(nn.Module):
    """Drop-in replacement for nn.Linear with int8 weights (per-channel scale).

    Each output channel (row of the weight matrix) has its own scale factor,
    which reduces quantization error compared to per-tensor — especially for
    layers with outlier weight values.
    """

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scales", scales)  # shape [out_features, 1]
        if bias is not None:
            self.bias = nn.Parameter(bias.clone().float())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> QuantizedLinearPerChannel:
        """Build a QuantizedLinearPerChannel from an existing nn.Linear."""
        q, scales = int8_quantize_per_channel(linear.weight.data)
        return cls(
            weight_int8=q,
            scales=scales,
            bias=linear.bias.data if linear.bias is not None else None,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp32 = self.weight_int8.float() * self.scales
        return F.linear(x, weight_fp32, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"dtype=int8, granularity=per_channel")


# ═══════════════════════════════════════════════════════════════════════════
# Selective model quantization
# ═══════════════════════════════════════════════════════════════════════════

# Layers whose names contain any of these substrings are NOT quantized,
# because they are either numerically sensitive (LayerNorm) or
# semantically important to keep at full precision (classification head).
SKIP_PATTERNS = ["norm", "cls_token", "pos_embed", "head"]


def _set_nested_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a submodule by its dotted name (e.g. 'blocks.0.attn.qkv')."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def quantize_model_selective(
    model: nn.Module,
    skip_patterns: list[str] | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Quantize all nn.Linear layers to INT8 per-tensor, skipping sensitive ones.

    Walks the model graph and replaces each nn.Linear whose name does NOT
    match any of the skip_patterns with a QuantizedLinear wrapper.

    Args:
        model:         The model to quantize (modified **in-place**).
        skip_patterns: Substrings to match against layer names.  Layers whose
                       name contains any pattern are left in float32.
                       Defaults to SKIP_PATTERNS (norm, cls_token, pos_embed, head).
        verbose:       If True, print per-layer quantization statistics.

    Returns:
        model:       The same model object, with Linear layers replaced.
        layer_stats: List of dicts with per-layer shape, MSE, MAE, scale.
    """
    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS

    layer_stats: list[dict] = []
    quantized: list[str] = []
    skipped: list[str] = []

    # Snapshot the list first to avoid mutating the dict during iteration
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    for name, module in targets:
        if any(p in name for p in skip_patterns):
            skipped.append(name)
            continue

        w = module.weight.data
        q, sc = int8_quantize(w)
        err = quantization_error(w, q, sc)

        _set_nested_module(model, name, QuantizedLinear.from_linear(module))

        layer_stats.append({"layer": name, "shape": list(w.shape),
                            "n_params": w.numel(), **err})
        quantized.append(name)

    if verbose:
        print(f"\n  Quantized layers ({len(quantized)}):")
        for s in layer_stats:
            print(f"    {s['layer']:<45}  shape={s['shape']}  "
                  f"MSE={s['mse']:.2e}  scale={s['scale']:.5f}")
        print(f"\n  Skipped layers ({len(skipped)}):")
        for name in skipped:
            print(f"    {name}")

    return model, layer_stats


def quantize_model_per_channel(
    model: nn.Module,
    skip_patterns: list[str] | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Same as quantize_model_selective but with per-channel granularity."""
    if skip_patterns is None:
        skip_patterns = SKIP_PATTERNS

    layer_stats: list[dict] = []
    quantized: list[str] = []
    skipped: list[str] = []

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    for name, module in targets:
        if any(p in name for p in skip_patterns):
            skipped.append(name)
            continue

        w = module.weight.data
        q, scales = int8_quantize_per_channel(w)
        err = quantization_error_per_channel(w, q, scales)

        _set_nested_module(model, name, QuantizedLinearPerChannel.from_linear(module))

        layer_stats.append({"layer": name, "shape": list(w.shape),
                            "n_params": w.numel(), **err})
        quantized.append(name)

    if verbose:
        print(f"\n  Quantized layers — per-channel ({len(quantized)}):")
        for s in layer_stats:
            print(f"    {s['layer']:<45}  shape={s['shape']}  MSE={s['mse']:.2e}")
        print(f"\n  Skipped layers ({len(skipped)}):")
        for name in skipped:
            print(f"    {name}")

    return model, layer_stats
