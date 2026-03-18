"""
INT8 cuantizare cu scalare liniară (per-tensor) pentru straturi nn.Linear.

Abordare:
  - scale = max(|W|) / 127
  - q = round(W / scale).clamp(-128, 127).to(int8)
  - dequantize la inferență: W_approx = q.float() * scale

Aceasta garantează o mapare bijectivă la 256 valori discrete, spre deosebire
de torch.float8_e4m3fn unde nu toate valorile intermediare sunt reprezentabile.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Funcții de cuantizare / decuantizare
# ---------------------------------------------------------------------------

def int8_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor linear quantization: float32 → int8 + scale.

    Returns:
        q     : torch.int8 tensor, same shape as input
        scale : float, multiply q by this to recover the original values
    """
    max_val = tensor.abs().max().item()
    scale   = max_val / 127.0 if max_val > 0 else 1.0
    q = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def int8_dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Recover float32 approximation from int8 + scale."""
    return q.float() * scale


def quantization_error(original: torch.Tensor, q: torch.Tensor, scale: float) -> dict:
    """Compute MSE and MAE between original tensor and its int8 reconstruction."""
    reconstructed = int8_dequantize(q, scale)
    diff = original.float() - reconstructed
    return {
        "mse":  diff.pow(2).mean().item(),
        "mae":  diff.abs().mean().item(),
        "max_abs_error": diff.abs().max().item(),
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Wrapper modul
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """Drop-in replacement pentru nn.Linear cu weight cuantizat la INT8.

    Ponderile sunt stocate ca int8; la fiecare forward pass sunt decuantizate
    la float32 înainte de înmulțirea matriceală.
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
        self.in_features  = in_features
        self.out_features = out_features
        self.scale        = scale

        # Stocăm ponderile ca buffer (nu parametru — nu se antrenează)
        self.register_buffer("weight_int8", weight_int8)

        if bias is not None:
            self.bias = nn.Parameter(bias.clone().float())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        """Construiește un QuantizedLinear dintr-un nn.Linear existent."""
        q, scale = int8_quantize(linear.weight.data)
        return cls(
            weight_int8=q,
            scale=scale,
            bias=linear.bias.data if linear.bias is not None else None,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decuantizare la inferență — overhead acceptabil pentru evaluare statică
        weight_fp32 = int8_dequantize(self.weight_int8, self.scale)
        return F.linear(x, weight_fp32, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"scale={self.scale:.6f}, dtype=int8"
        )


# ---------------------------------------------------------------------------
# Cuantizare selectivă a modelului
# ---------------------------------------------------------------------------

# Straturi excluse de la cuantizare (sensibile la erori numerice sau semantic importante)
SKIP_PATTERNS = ["norm", "cls_token", "pos_embed", "head"]


def _set_nested_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Setează un sub-modul după nume (ex. 'blocks.0.attn.qkv')."""
    parts  = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def quantize_model_selective(
    model: nn.Module,
    skip_patterns: list[str] = SKIP_PATTERNS,
    verbose: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """Cuantizează selectiv toate nn.Linear din Attention + MLP.

    Sare peste straturi ale căror nume conțin oricare din `skip_patterns`.

    Returns:
        model      : modelul modificat in-place (ponderile înlocuite cu QuantizedLinear)
        layer_stats: listă de dict cu statistici per strat cuantizat
    """
    layer_stats   = []
    quantized     = []
    skipped       = []

    # Colectăm lista mai întâi (evităm modificarea dicționarului în iterație)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    for name, module in targets:
        if any(p in name for p in skip_patterns):
            skipped.append(name)
            continue

        # Calculăm statistici înainte de cuantizare
        w     = module.weight.data
        q, sc = int8_quantize(w)
        err   = quantization_error(w, q, sc)

        # Înlocuim modulul
        q_linear = QuantizedLinear.from_linear(module)
        _set_nested_module(model, name, q_linear)

        stats = {
            "layer":    name,
            "shape":    list(w.shape),
            "n_params": w.numel(),
            **err,
        }
        layer_stats.append(stats)
        quantized.append(name)

    if verbose:
        print(f"\n  Straturi cuantizate ({len(quantized)}):")
        for s in layer_stats:
            print(f"    {s['layer']:<45}  "
                  f"shape={s['shape']}  "
                  f"MSE={s['mse']:.2e}  "
                  f"scale={s['scale']:.5f}")
        print(f"\n  Straturi sărite ({len(skipped)}):")
        for name in skipped:
            print(f"    {name}")

    return model, layer_stats
