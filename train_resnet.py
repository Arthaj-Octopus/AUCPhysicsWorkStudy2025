import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.models as models
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -------------------------
# Load Data
# -------------------------
with open("combined_scattering_data.pkl", "rb") as f:
    df = pickle.load(f)

print("DataFrame columns:", df.columns)
print("Class distribution:", df['particle_type'].value_counts())

# Convert scattering patterns to tensor
X = np.stack(df["scattering_pattern"].to_numpy())
y = df['particle_type'].map({'plastic': 0, 'colloid': 1}).values

# Add and replicate channel dimension
if X.ndim == 3:
    X = X[:, np.newaxis, :, :]
if X.shape[1] == 1:
    X = np.repeat(X, 3, axis=1)

# Use ImageNet normalization for pretrained model
X = X.astype(np.float32) / 255.0  # Scale to [0, 1]
X = (X - np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# -------------------------
# Train/Validation split with stratification
# -------------------------
indices = np.arange(len(X))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)

train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

# Use smaller batch size for CPU
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# -------------------------
# Simplified ResNet Model for CPU
# -------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')

        # Freeze early layers to reduce computation
        for param in list(self.resnet.parameters())[:-30]:  # Freeze more layers for CPU
            param.requires_grad = False

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Simplified classifier for CPU
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        return x

num_classes = 2
model = ResNetClassifier(num_classes)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

# Class-weighted loss to handle imbalance
class_counts = np.bincount(y)
class_weights = torch.tensor([1.0, class_counts[0]/class_counts[1]], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer with different learning rates
optimizer = optim.Adam([
    {'params': model.resnet.parameters(), 'lr': 0.00001},
    {'params': model.classifier.parameters(), 'lr': 0.0001}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# -------------------------
# Training with Early Stopping
# -------------------------
num_epochs = 20  # Reduced for CPU
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_loss = float('inf')
patience_counter = 0
patience = 5  # Reduced patience for CPU

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress more frequently
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Validation phase
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    # Calculate metrics
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    train_acc = 100.0 * correct / total
    val_acc = 100.0 * val_correct / val_total

    # Update learning rate
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved with val loss: {avg_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Store metrics
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# -------------------------
# Final Evaluation
# -------------------------
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['plastic', 'colloid'],
            yticklabels=['plastic', 'colloid'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['plastic', 'colloid']))

# -------------------------
# Plot Results
# -------------------------
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over Epochs")

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy over Epochs")

plt.tight_layout()
plt.savefig('training_results.png')

plt.show()
