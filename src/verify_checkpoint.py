from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_FILE = PROJECT_ROOT / "checkpoints" / "baseline_cnn_best.pth"


def main():
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_FILE}")

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
    )

    print("Checkpoint Verification")
    print("-----------------------")

    print(f"Epoch              : {checkpoint['epoch']}")
    print(f"Validation Accuracy: {checkpoint['validation_accuracy']:.4f}")
    print(f"Architecture       : {checkpoint['architecture']}")
    print(f"Number of Classes  : {checkpoint['num_classes']}")
    print(f"Image Size         : {checkpoint['image_size']}")
    print(f"Batch Size         : {checkpoint['batch_size']}")
    print(f"Optimizer          : {checkpoint['optimizer']}")
    print(f"Learning Rate      : {checkpoint['learning_rate']}")
    print(f"Loss Function      : {checkpoint['loss_function']}")
    print(f"Seed               : {checkpoint['seed']}")

    print("\nClass Mapping")
    print("-------------")

    for class_name, class_index in checkpoint["class_mapping"].items():
        print(f"{class_index}: {class_name}")

    print("\nCheckpoint verification: PASS")


if __name__ == "__main__":
    main()
