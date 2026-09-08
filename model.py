import torch
import torch.nn as nn


class MicroCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Vision encoder
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        # Flattened image → 128 features
        self.fc = nn.Linear(8 * 32 * 32, 128)

        # 3 colours × 2 coordinates = 6 numbers
        self.position = nn.Linear(128, 6)

    def forward(self, x):

        # Look at image
        x = self.conv(x)

        # Activation
        x = self.relu(x)

        # Flatten
        x = x.flatten(1)

        # 8192 → 128
        x = self.fc(x)

        x = self.relu(x)

        # 128 → 6
        x = self.position(x)

        # [batch, 6] → [batch, 3, 2]
        x = x.reshape(-1, 3, 2)

        return x


model = MicroCNN()

x = torch.randn(1, 3, 32, 32)

y = model(x)

print("Input:", x.shape)
print("Output:", y.shape)