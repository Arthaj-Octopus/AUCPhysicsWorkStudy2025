import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image

# ==== Load Data ====
df = pd.read_pickle(r"D:\Resnet\combined_scattering_data.pkl")

# Map particle_type to integers
label_to_idx = {label: idx for idx, label in enumerate(df['particle_type'].unique())}
df['label_idx'] = df['particle_type'].map(label_to_idx)

# ==== Custom Dataset ====
class ScatteringDataset(Dataset):
    def _init_(self, dataframe, transform=None):  # double underscores
        self.dataframe = dataframe
        self.transform = transform

    def _len_(self):  # double underscores
        return len(self.dataframe)

    def _getitem_(self, idx):  # double underscores
        pattern = self.dataframe.iloc[idx]['scattering_pattern']  # 2D array
        label = self.dataframe.iloc[idx]['label_idx']

        # Convert to PIL Image and duplicate to 3 channels
        img = Image.fromarray(pattern.astype(np.float32))
        img = img.convert("RGB")  # makes 3 identical channels

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==== Train/Test Split ====
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

train_dataset = ScatteringDataset(train_df, transform=transform)
val_dataset = ScatteringDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==== Model ====
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze feature extractor

num_classes = len(label_to_idx)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ==== Training Setup ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ==== Training Loop ====
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10] Loss: {running_loss/len(train_loader):.4f}")

# ==== Validation ====
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# ==== Save Model ====
torch.save(model.state_dict(), r"D:\Resnet\resnet_model.pth")