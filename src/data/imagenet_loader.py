"""
ImageNet-1k validation loader — shared across all evaluation scripts (Phases 1-3).

Loads from local parquet files first (``data/imagenet-1k/*.parquet``),
falls back to HuggingFace streaming if not found.
"""

from pathlib import Path

from torch.utils.data import DataLoader, Dataset


_DEFAULT_DATA_DIR = "data/imagenet-1k"


class HFImageNet(Dataset):
    """Thin wrapper around a HuggingFace dataset with an image transform."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_imagenet_val(
    transform,
    batch_size: int = 64,
    num_workers: int = 2,
    data_dir: str = _DEFAULT_DATA_DIR,
) -> DataLoader:
    """Load ImageNet-1k validation split from local parquet files or HuggingFace.

    Args:
        transform:   torchvision-compatible image transform.
        batch_size:  DataLoader batch size.
        num_workers: DataLoader worker count.
        data_dir:    Path to folder containing ``*.parquet`` files.

    Returns:
        A ``DataLoader`` yielding ``(images, labels)`` batches.
    """
    from datasets import load_dataset

    local_path = Path(data_dir)
    if local_path.exists() and any(local_path.glob("*.parquet")):
        print(f"Loading ImageNet-1k validation from local parquet: {local_path}")
        hf_dataset = load_dataset(
            "parquet",
            data_files=str(local_path / "*.parquet"),
            split="train",  # parquet files load as "train" split
        )
    else:
        print("Loading ImageNet-1k validation split from HuggingFace ...")
        hf_dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            trust_remote_code=True,
        )
    print(f"  {len(hf_dataset)} images\n")

    dataset = HFImageNet(hf_dataset, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
