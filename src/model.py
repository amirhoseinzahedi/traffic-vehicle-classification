import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Simple CNN baseline for 8-class vehicle classification."""

    def __init__(self, num_classes=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    """Run basic model checks."""

    model = BaselineCNN(num_classes=8)

    print("Model")
    print("-----")
    print(model)

    dummy_input = torch.randn(32, 3, 128, 128)

    output = model(dummy_input)

    print("\nTensor Shapes")
    print("-------------")
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"\nTrainable parameters: {trainable_parameters:,}")


if __name__ == "__main__":
    main()
