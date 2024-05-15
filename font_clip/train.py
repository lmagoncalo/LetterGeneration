import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import FontCLIPModel

# Load data from the numpy file
data = np.load("font_clip_data.npz")
images = data['images']
input_ids = data['input_ids']
attention_masks = data['attention_masks']

print(f"Number of images: {len(images)}, Images Shape: {images.shape}, Input IDS Shape: {input_ids.shape}, Attention Masks Shape: {attention_masks.shape}")

# Split data into training and validation sets
images_train, images_val, input_ids_train, input_ids_val, attention_masks_train, attention_masks_val = train_test_split(images, input_ids, attention_masks, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
images_train_tensor = torch.tensor(images_train, dtype=torch.float32)
images_val_tensor = torch.tensor(images_val, dtype=torch.float32)

input_ids_train_tensor = torch.tensor(input_ids_train, dtype=torch.long).squeeze(1)
input_ids_val_tensor = torch.tensor(input_ids_val, dtype=torch.long).squeeze(1)

attention_masks_train_tensor = torch.tensor(attention_masks_train, dtype=torch.long).squeeze(1)
attention_masks_val_tensor = torch.tensor(attention_masks_val, dtype=torch.long).squeeze(1)

# Create PyTorch datasets
train_dataset = TensorDataset(images_train_tensor, input_ids_train_tensor, attention_masks_train_tensor)
val_dataset = TensorDataset(images_val_tensor, input_ids_val_tensor, attention_masks_val_tensor)

# Define batch size
batch_size = 64

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = FontCLIPModel().to(device)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    if epoch % 4 == 0:
        for g in optimizer.param_groups:
            g['lr'] *= 0.1

    model.train()  # Set model to training mode
    train_loss = 0.0
    for images, input_ids, attention_masks in train_loader:
        images, input_ids, attention_masks = images.to(device), input_ids.to(device), attention_masks.to(device)

        optimizer.zero_grad()
        loss = model(images, input_ids, attention_masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Calculate average training loss and accuracy
    train_loss = train_loss / len(train_dataset)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images, input_ids, attention_masks in val_loader:
            images, input_ids, attention_masks = images.to(device), input_ids.to(device), attention_masks.to(device)

            loss = model(images, input_ids, attention_masks)
            val_loss += loss.item()

    # Calculate average validation loss and accuracy
    val_loss = val_loss / len(val_dataset)

    # Log results
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "font_clip_model.pth")