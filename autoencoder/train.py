import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn.functional as F

from models import Autoencoder


# Define transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
# Load data from the numpy file
data = np.load("font_data.npz")
images = data['images']

print(f"Number of images: {len(images)}, Images Shape: {images.shape}")

# Split data into training and validation sets
X_train, X_val = train_test_split(images, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor)
val_dataset = TensorDataset(X_val_tensor)

# Define batch size
batch_size = 128

# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize the autoencoder
# model = VectorQuantizedVAE(3, 64, 128).to(device)
model = Autoencoder().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# Train the autoencoder
num_epochs = 20
for epoch in tqdm(range(num_epochs + 1)):
    for images, in train_loader:
        images = images.to(device)

        optimizer.zero_grad()

        x_recons = model(images)

        # Reconstruction loss
        loss = F.mse_loss(x_recons, images)

        loss.backward()

        optimizer.step()

    if epoch % 5 == 0:
        with torch.no_grad():
            loss = 0.
            for images,  in val_loader:
                images = images.to(device)
                x_recons = model(images)
                loss += F.mse_loss(x_recons, images)

            loss /= len(val_loader)

        fixed_images, = next(iter(val_loader))
        with torch.no_grad():
            images = images.to(device)
            x_recons = model(images)
            save_image(x_recons, 'recons_epoch_{}.png'.format(epoch))

        print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item(), loss.item()))

# Save the model
torch.save(model.state_dict(), 'conv_autoencoder.pth')