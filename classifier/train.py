import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import *
from tqdm import tqdm

# Load data from the numpy file
data = np.load("font_data.npz")
images = data['images']
labels = data['labels']

print(f"Number of images: {len(images)}, Images Shape: {images.shape}, Labels Shape: {labels.shape}")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Define batch size
batch_size = 64

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)  # Adjust num_classes as needed
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=5)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    if epoch % 4 == 0:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1

    model.train()  # Set model to training mode
    train_loss = 0.0
    train_correct = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == targets).sum().item()

    # Calculate average training loss and accuracy
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_correct / len(train_dataset)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == targets).sum().item()

    # Calculate average validation loss and accuracy
    val_loss = val_loss / len(val_dataset)
    val_accuracy = val_correct / len(val_dataset)

    # Log results
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "style_classifier.pth")