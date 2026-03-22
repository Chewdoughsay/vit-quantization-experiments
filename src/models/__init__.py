"""
src.models — ViT model creation and INT8 weight-only quantization.

Public API:
    create_vit_model          Create a ViT variant via timm (Tiny / Small / Base).
    count_parameters          Return (total, trainable) parameter counts.
    get_model_info            Parameter counts as a dict (absolute + millions).

    QuantizedLinear           Drop-in nn.Linear replacement (per-tensor INT8).
    QuantizedLinearPerChannel Drop-in nn.Linear replacement (per-channel INT8).
    quantize_model_selective  Quantize all Linear layers except norm/head (per-tensor).
    quantize_model_per_channel Same, but per-channel granularity.
"""

from src.models.vit_model import create_vit_model, count_parameters, get_model_info
from src.models.quantized_linear import (
    QuantizedLinear,
    QuantizedLinearPerChannel,
    quantize_model_selective,
    quantize_model_per_channel,
)
