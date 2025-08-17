import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Load DataFrame from pickle
# -------------------------
with open("combined_scattering_data.pkl", "rb") as f:
    df = pickle.load(f)

print("DataFrame columns:", df.columns)

# Convert scattering patterns to a proper tensor
X = torch.tensor(np.stack(df["scattering_pattern"].to_numpy()), dtype=torch.float32)

# Replace 'label' with the actual column name in your dataset (e.g., 'particle_type')
y = torch.tensor(pd.factorize(df["particle_type"])[0], dtype=torch.long)

# Add channel dimension if missing (N, 227, 227) → (N, 1, 227, 227)
if X.ndim == 3:
    X = X.unsqueeze(1)

# -------------------------
# Train/Validation split
# -------------------------
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------------
# Define ResNet-18 Model
# -------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(
            X.shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

num_classes = len(torch.unique(y))
model = ResNetClassifier(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# Training Loop with Tracking
# -------------------------
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(10):
    # Training
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/10] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# -------------------------
# Plot Loss & Accuracy
# -------------------------
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over Epochs")

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy over Epochs")

plt.show()
