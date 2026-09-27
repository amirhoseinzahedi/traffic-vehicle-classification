from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
UNCLEAN_DIR = DATASET_DIR / "unclean"

SPLIT_DIR = PROJECT_ROOT / "configs"
VALIDATION_INDICES_FILE = SPLIT_DIR / "validation_indices.npy"


# Reproducibility settings
SEED = 42
VALIDATION_SIZE = 0.20


def load_datasets():
    """Load train, test, and unclean datasets separately."""

    train_dataset = datasets.ImageFolder(TRAIN_DIR)
    test_dataset = datasets.ImageFolder(TEST_DIR)
    unclean_dataset = datasets.ImageFolder(UNCLEAN_DIR)

    return train_dataset, test_dataset, unclean_dataset


def verify_class_mappings(train_dataset, test_dataset):
    """Verify that train and test use the same class mapping."""

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train and test class mappings are different.")


def create_validation_split(train_dataset):
    """Create a reproducible stratified train/validation split."""

    indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.targets)

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    return train_indices, validation_indices


def save_validation_indices(validation_indices):
    """Save validation indices for reproducibility."""

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        VALIDATION_INDICES_FILE,
        validation_indices,
    )


def print_split_summary(
    train_dataset,
    train_indices,
    validation_indices,
):
    """Print dataset and split information."""

    print("\nDataset Split Summary")

    print(f"Total training images : {len(train_dataset)}")
    print(f"Training subset       : {len(train_indices)}")
    print(f"Validation subset     : {len(validation_indices)}")

    print("\nClass Mapping")

    for class_name, class_index in train_dataset.class_to_idx.items():
        print(f"{class_index}: {class_name}")

    print("\nSplit Configuration")

    print(f"Seed                  : {SEED}")
    print(f"Validation size       : {VALIDATION_SIZE}")
    print(f"Validation indices    : {VALIDATION_INDICES_FILE}")


def verify_class_distribution(train_dataset, train_indices, validation_indices):
    """Verify that the stratified split preserves class proportions."""
    train_labels = np.array(train_dataset.targets)

    print("\nClass Distribution")

    for class_name, class_index in train_dataset.class_to_idx.items():
        train_count = np.sum(train_labels[train_indices] == class_index)
        validation_count = np.sum(train_labels[validation_indices] == class_index)

        print(
            f"{class_name:10s} | "
            f"train: {train_count:2d} | "
            f"validation: {validation_count:2d}"
        )

        if train_count != 40 or validation_count != 10:
            raise ValueError(f"Unexpected class distribution for {class_name}.")


def verify_reproducibility(train_dataset, validation_indices):
    """Verify that the same seed reproduces the same split."""
    indices = np.arange(len(train_dataset))
    labels = np.array(train_dataset.targets)

    _, second_validation_indices = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    if not np.array_equal(validation_indices, second_validation_indices):
        raise ValueError("Validation split is not reproducible.")

    print("\nReproducibility Check")
    print("Validation indices are reproducible: PASS")


def main():
    train_dataset, test_dataset, unclean_dataset = load_datasets()

    verify_class_mappings(train_dataset, test_dataset)

    train_indices, validation_indices = create_validation_split(train_dataset)

    save_validation_indices(validation_indices)

    print_split_summary(
        train_dataset,
        train_indices,
        validation_indices,
    )

    verify_class_distribution(
        train_dataset,
        train_indices,
        validation_indices,
    )

    verify_reproducibility(
        train_dataset,
        validation_indices,
    )

    print("\nTest Set")
    print(f"Test images           : {len(test_dataset)}")

    print("\nUnclean Set")
    print(f"Unclean images        : {len(unclean_dataset)}")
    print(f"Unclean classes       : {unclean_dataset.classes}")


if __name__ == "__main__":
    main()
