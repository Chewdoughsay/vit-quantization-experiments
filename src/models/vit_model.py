"""ViT model creation utilities using the timm library."""
import timm
import torch
import torch.nn as nn


def create_vit_model(model_name='vit_tiny_patch16_224', num_classes=10, pretrained=False):
    """Create a Vision Transformer model from timm."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    
    return model


def count_parameters(model):
    """Return (total_params, trainable_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_info(model):
    """Return dict with parameter counts (total and trainable) in units and millions."""
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6,
    }
    
    return info


# Predefined model configurations for easy reference
MODEL_CONFIGS = {
    'vit_tiny': {
        'name': 'vit_tiny_patch16_224',
        'params_approx': '5.7M',
        'description': 'ViT-Tiny - smallest model, good for fast testing'
    },
    'vit_small': {
        'name': 'vit_small_patch16_224',
        'params_approx': '22M',
        'description': 'ViT-Small - medium model, good compromise'
    },
    'vit_base': {
        'name': 'vit_base_patch16_224',
        'params_approx': '86M',
        'description': 'ViT-Base - standard model, larger size'
    },
}


if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Test models
    print("=== Testing ViT Models ===\n")
    
    for key, config in MODEL_CONFIGS.items():
        print(f"{key.upper()}:")
        model = create_vit_model(config['name'], num_classes=10, pretrained=False)
        info = get_model_info(model)
        print(f"  Parameters: {info['trainable_params_millions']:.2f}M")
        print(f"  Description: {config['description']}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")  # Should be [2, 10]
        print()