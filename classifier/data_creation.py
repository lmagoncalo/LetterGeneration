import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from tqdm import tqdm

# Define the root directory where your font dataset is stored
root_dir = "dataset/"

# Define the maximum number of classes
max_classes = 3

# Create an empty list to store images and labels
data = []

# Map category names to integer class labels
class_label_mapping = {}

# Function to extract category from metadata.pb file
def get_category_from_pb(metadata_path):
    # Open the metadata.pb file and extract the category information
    with open(metadata_path, 'r') as f:
        for line in f:
            if line.startswith("category:"):
                category = line.split(":")[1].strip()
                return category

# Iterate over folders
for category_folder in tqdm(os.listdir(root_dir)):
    category_dir = os.path.join(root_dir, category_folder)
    metadata_path = os.path.join(category_dir, 'METADATA.pb')
    if os.path.exists(metadata_path):
        # Get category name from metadata.pb file
        category = get_category_from_pb(metadata_path)
        # Map category names to integer labels
        if category not in class_label_mapping:
            label = len(class_label_mapping)
            class_label_mapping[category] = label
            # Create a folder for the category
            os.makedirs(os.path.join("images/", str(label)), exist_ok=True)

        class_label = class_label_mapping[category]
        # Find font files in the category folder
        font_files = [file for file in os.listdir(category_dir) if file.endswith('.ttf') or file.endswith('.otf')]
        # Generate images for each font file and letter
        for font_file in font_files:
            font_path = os.path.join(category_dir, font_file)
            try:
                font = ImageFont.truetype(font_path, size=60)
                for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    img = Image.new('RGB', (64, 64), color='white')
                    draw = ImageDraw.Draw(img)
                    left, top, right, bottom = draw.textbbox((0, 0), letter, font=font)
                    x = (64 - (right - left)) // 2
                    y = (32 - (bottom - top)) // 2
                    draw.text((x, y), letter, fill='black', font=font)
                    # img_tensor = TF.pil_to_tensor(img) / 255
                    # img_np = img_tensor.numpy()
                    # save_image(img_tensor, f"images/temp.png")
                    img.save(f"images/{class_label}/{category_folder}_{letter}.png")
                    # Append image and corresponding label to the data list
                    # data.append((img_np, class_label))
            except Exception as e:
                print(f"Error processing {font_path}: {e}")


"""
# Shuffle the data
np.random.shuffle(data)

# Split data into images and labels
images = np.array([d[0] for d in data])
labels = np.array([d[1] for d in data])

print(f"Number of images: {len(images)}, Shape: {images.shape}")
print(f"Number of labels: {len(labels)}, Shape: {labels.shape}")

print(class_label_mapping)

temp_images = torch.tensor(images[:16], dtype=torch.float)
temp_images = temp_images.view(-1, 3, 64, 64)

print(temp_images.shape)

save_image(temp_images, "images_end.png")

# Save the data as a numpy file
np.savez("font_data.npz", images=images, labels=labels)
"""
