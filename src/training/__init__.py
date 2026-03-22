"""
src.training — ViT training loop with AMP and monitoring.  **Legacy** (preliminary study only).

This package is used exclusively by scripts/preliminary/train.py for the
initial CIFAR-10 fine-tuning experiments.  The main quantization pipeline
(Phases 1-3) performs inference-only evaluation and does not use this module.
"""
