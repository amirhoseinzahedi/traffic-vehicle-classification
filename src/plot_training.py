from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_FILE = PROJECT_ROOT / "reports" / "baseline_history.csv"
PLOTS_DIR = PROJECT_ROOT / "reports" / "figures"


def load_history():
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"Training history not found: {HISTORY_FILE}")

    return pd.read_csv(HISTORY_FILE)


def plot_loss(history):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline CNN - Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = PLOTS_DIR / "baseline_loss_curve.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_accuracy(history):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o",
        label="Train Accuracy",
    )

    plt.plot(
        history["epoch"],
        history["validation_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline CNN - Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = PLOTS_DIR / "baseline_accuracy_curve.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_learning_rate(history):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["learning_rate"],
        marker="o",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Baseline CNN - Learning Rate")
    plt.grid(True)
    plt.tight_layout()

    output_path = PLOTS_DIR / "baseline_learning_rate.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    history = load_history()

    loss_path = plot_loss(history)
    accuracy_path = plot_accuracy(history)
    learning_rate_path = plot_learning_rate(history)

    print("Training curves generated")
    print("-------------------------")
    print(f"Loss curve       : {loss_path}")
    print(f"Accuracy curve   : {accuracy_path}")
    print(f"Learning rate    : {learning_rate_path}")


if __name__ == "__main__":
    main()
