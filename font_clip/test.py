import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import torch.nn.functional as F

from font_clip.models import FontCLIPModel

from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load data from the numpy file
data = np.load("font_clip_data.npz")
images = data['images']
input_ids = data['input_ids']
attention_masks = data['attention_masks']

print(f"Number of images: {len(images)}, Images Shape: {images.shape}, Input IDS Shape: {input_ids.shape}, Attention Masks Shape: {attention_masks.shape}")

# Convert data to PyTorch tensors
images_tensor = torch.tensor(images, dtype=torch.float32)
input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).squeeze(1)
attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long).squeeze(1)

# Create PyTorch datasets
dataset = TensorDataset(images_tensor, input_ids_tensor, attention_masks_tensor)

# Define batch size
batch_size = 16

# Create DataLoader for training and validation sets
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = FontCLIPModel().to(device)
model.load_state_dict(torch.load("font_clip_model.pth"))

images, input_ids, attention_masks = next(iter(data_loader))

print(f"Number of images: {len(images)}, Images Shape: {images.shape}, Input IDS Shape: {input_ids.shape}, Attention Masks Shape: {attention_masks.shape}")

save_image(images, "image.png")

target = 2

text_decoded_original = tokenizer.decode(list(input_ids[0]))
text_decoded_target = tokenizer.decode(list(input_ids[target]))

print("Original:", text_decoded_original)
print("Target:", text_decoded_target)

image = images[0].unsqueeze(0).to(device)
text_target = input_ids[target].unsqueeze(0).to(device)
mask_target = attention_masks[target].unsqueeze(0).to(device)
text_original = input_ids[0].unsqueeze(0).to(device)
mask_original = attention_masks[0].unsqueeze(0).to(device)

print(image.shape, text_target.shape, mask_target.shape, text_original.shape, mask_original.shape)

image_embedding = model.encode_image(image)
text_embedding_target = model.encode_text(text_target, mask_target)
text_embedding_original = model.encode_text(text_original, mask_original)

print(image_embedding.shape, text_embedding_target.shape, text_embedding_original.shape)

image_embeddings_n = F.normalize(image_embedding, p=2, dim=-1)
text_embeddings_target_n = F.normalize(text_embedding_target, p=2, dim=-1)
text_embeddings_original_n = F.normalize(text_embedding_original, p=2, dim=-1)
dot_similarity_target = text_embeddings_target_n @ image_embeddings_n.T
dot_similarity_original = text_embeddings_original_n @ image_embeddings_n.T

print("Target:", dot_similarity_target.item(), "Original:", dot_similarity_original.item())
