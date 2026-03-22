"""
CIFAR-10 data loaders with basic and extended augmentation for 224×224 ViT input.

**Legacy** — used only by ``scripts/preliminary/train.py`` for the initial
CIFAR-10 fine-tuning study.  The main quantization pipeline (Phases 1-3)
downloads ImageNette directly inside each evaluation script.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


def get_project_root():
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'


def get_cifar10_loaders(
    batch_size=128,
    num_workers=2,
    augmentation='basic',
    data_dir=None,
    pin_memory=False
):
    """Return (train_loader, test_loader) for CIFAR-10 resized to 224×224."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    if augmentation == 'extended':
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(p=0.5),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(15),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),

            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value='random'
            ),
        ])
    else:  # basic
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])

    train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, test_loader


def get_dataset_info():
    """Return CIFAR-10 metadata dict (classes, sizes, channels)."""
    return {
        'name': 'CIFAR-10',
        'num_classes': 10,
        'train_size': 50000,
        'test_size': 10000,
        'image_size': (32, 32),
        'num_channels': 3,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    }


if __name__ == '__main__':
    # Test the loader
    print("Testing CIFAR-10 loaders...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DEFAULT_DATA_DIR}")

    # Test basic augmentation
    print("\n1. Basic augmentation:")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=32,
        augmentation='basic'
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test extended augmentation
    print("\n2. Extended augmentation:")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=32,
        augmentation='extended'
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\n3. Batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Dataset info
    print("\n4. Dataset info:")
    info = get_dataset_info()
    for key, value in info.items():
        print(f"   {key}: {value}")