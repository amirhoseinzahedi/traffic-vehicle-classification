from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

VALIDATION_INDICES_FILE = PROJECT_ROOT / "configs" / "validation_indices.npy"


# Reproducibility
SEED = 42

# DataLoader configuration
BATCH_SIZE = 32
NUM_WORKERS = 0

# Image configuration
IMAGE_SIZE = 128


def get_transforms():
    """Create training and evaluation transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    return train_transform, eval_transform


def load_datasets():
    """Load the training dataset twice for separate transforms."""

    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform,
    )

    validation_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    return train_dataset, validation_dataset, test_dataset


def verify_class_mapping(train_dataset, validation_dataset, test_dataset):
    """Verify that all datasets use the same class mapping."""

    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise ValueError("Training and validation class mappings are different.")

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Training and test class mappings are different.")


def create_train_validation_subsets(train_dataset, validation_dataset):
    """Create subsets using the fixed validation indices."""

    validation_indices = np.load(VALIDATION_INDICES_FILE)

    all_indices = np.arange(len(train_dataset))

    train_indices = np.setdiff1d(all_indices, validation_indices)

    train_subset = Subset(
        train_dataset,
        train_indices,  # type: ignore
    )

    validation_subset = Subset(validation_dataset, validation_indices)

    return train_subset, validation_subset


def create_dataloaders(train_subset, validation_subset, test_dataset):
    """Create DataLoaders for training, validation, and test."""

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, validation_loader, test_loader


def main():
    """Run dataset pipeline checks."""

    train_dataset, validation_dataset, test_dataset = load_datasets()

    verify_class_mapping(
        train_dataset,
        validation_dataset,
        test_dataset,
    )

    train_subset, validation_subset = create_train_validation_subsets(
        train_dataset,
        validation_dataset,
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_subset,
        validation_subset,
        test_dataset,
    )

    print("Dataset Pipeline")
    print("----------------")
    print(f"Classes       : {train_dataset.classes}")
    print(f"Train samples : {len(train_subset)}")
    print(f"Val samples   : {len(validation_subset)}")
    print(f"Test samples  : {len(test_dataset)}")

    print("\nDataLoader")
    print("----------")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(validation_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print("\nFirst Training Batch")
    print("--------------------")
    print(f"Image tensor shape : {images.shape}")
    print(f"Label tensor shape : {labels.shape}")
    print(f"Image dtype        : {images.dtype}")
    print(f"Label dtype        : {labels.dtype}")


if __name__ == "__main__":
    main()
