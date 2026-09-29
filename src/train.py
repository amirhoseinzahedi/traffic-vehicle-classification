from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from dataset import create_train_validation_subsets
from dataset import load_datasets
from dataset import verify_class_mapping

from model import BaselineCNN  # from model.py

import csv


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Training configuration
SEED = 42
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

BATCH_SIZE = 32


REPORT_DIR = PROJECT_ROOT / "reports"
HISTORY_FILE = REPORT_DIR / "baseline_history.csv"


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_function, optimizer):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, loader, loss_function):
    """Evaluate the model on the validation set."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model,
    epoch,
    validation_accuracy,
    class_mapping,
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / "baseline_cnn_best.pth"

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_accuracy": validation_accuracy,
        "class_mapping": class_mapping,
        "architecture": "BaselineCNN",
        "num_classes": len(class_mapping),
        "image_size": 128,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss_function": "CrossEntropyLoss",
        "seed": SEED,
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def save_history(history):
    """Save training history as a CSV file."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
                "learning_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(history)


def main():
    """Train the baseline CNN."""

    set_seed(SEED)

    print(f"Device: {DEVICE}")

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

    model = BaselineCNN(num_classes=len(train_dataset.classes)).to(DEVICE)

    class_mapping = train_dataset.class_to_idx

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []
    best_validation_accuracy = 0.0
    best_validation_loss = float("inf")

    print("\nTraining")
    print("--------")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_function, optimizer
        )

        validation_loss, validation_accuracy = validate(
            model, validation_loader, loss_function
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": current_learning_rate,
            }
        )

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy:.4f} "
            f"| Val Loss: {validation_loss:.4f} "
            f"| Val Acc: {validation_accuracy:.4f}"
            f"| LR: {current_learning_rate:.6f}"
        )

        if validation_accuracy > best_validation_accuracy or (
            validation_accuracy == best_validation_accuracy
            and validation_loss < best_validation_loss
        ):
            best_validation_accuracy = validation_accuracy

            best_validation_loss = validation_loss

            checkpoint_path = save_checkpoint(
                model,
                epoch + 1,
                validation_accuracy,
                class_mapping,
            )

            print(f"  Best checkpoint saved: {checkpoint_path}")

    save_history(history)

    print(f"\nTraining history saved: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
