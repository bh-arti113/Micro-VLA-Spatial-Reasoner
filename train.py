import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MicroVLADataset
from model import MicroCNN


# Dataset
dataset = MicroVLADataset(1000)

# Batches
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# Model
model = MicroCNN()


# Loss function
loss_fn = nn.MSELoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
for epoch in range(30):

    total_loss = 0

    for images, targets in loader:

        # Prediction
        predictions = model(images)

        # Compare prediction with real answer
        loss = loss_fn(predictions, targets)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Adjust weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        "Epoch:",
        epoch + 1,
        "Loss:",
        average_loss
    )


# Save trained model
torch.save(
    model.state_dict(),
    "microcnn.pth"
)

print("Model saved!")