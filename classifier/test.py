import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v2
from torchvision.utils import save_image

# Load data from the numpy file
data = np.load("font_data.npz")
images = data['images']
labels = data['labels']

print(f"Number of images: {len(images)}, Images Shape: {images.shape}, Labels Shape: {labels.shape}")

# Convert data to PyTorch tensors
images = torch.tensor(images, dtype=torch.float32)  # Normalize pixel values to [0, 1]
images = images.view(-1, 3, 64, 64)
labels = torch.tensor(labels, dtype=torch.long)

# Create PyTorch datasets
dataset = TensorDataset(images, labels)

# Define batch size
batch_size = 16

# Create DataLoader for training and validation sets
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = mobilenet_v2()  # Adjust num_classes as needed
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
model.to(device)

model.load_state_dict(torch.load("style_classifier.pth"))

images, labels = next(iter(data_loader))

print(f"Images Shape: {images.shape}, Labels Shape: {labels.shape}")

save_image(images, "image.png")

images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    print(f"Predicted: {predicted.cpu().numpy()}")
    print(f"Labels:    {labels.numpy()}")

