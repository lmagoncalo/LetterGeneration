import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from tqdm import tqdm

# Define the root directory where your font dataset is stored
root_dir = "../classifier/dataset/"

# Create an empty list to store images and labels
data = []

# Iterate over folders
for category_folder in tqdm(os.listdir(root_dir)):
    category_dir = os.path.join(root_dir, category_folder)
    # Find font files in the category folder
    font_files = [file for file in os.listdir(category_dir) if file.endswith('.ttf') or file.endswith('.otf')]
    # Generate images for each font file and letter
    for font_file in font_files:
        font_path = os.path.join(category_dir, font_file)
        try:
            font = ImageFont.truetype(font_path, size=90)
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                img = Image.new('RGB', (128, 128), color='white')
                draw = ImageDraw.Draw(img)
                left, top, right, bottom = draw.textbbox((0, 0), letter, font=font, align="center")
                w = right - left
                h = bottom - top
                y = -top + (64 - (h // 2))
                x = -left + (64 - (w // 2))
                draw.text((x, y), letter, fill='black', font=font)
                img_tensor = TF.pil_to_tensor(img) / 255
                img_np = img_tensor.numpy()
                # save_image(img_tensor, f"images/temp.png")
                # img.save(f"images/{class_label}/{category_folder}_{letter}.png")
                # Append image and corresponding label to the data list
                data.append(img_np)
        except Exception as e:
            print(f"Error processing {font_path}: {e}")

# Shuffle the data
np.random.shuffle(data)

# Split data into images and labels
images = np.stack(data)

print(f"Number of images: {len(images)}, Shape: {images.shape}")

temp_images = torch.tensor(images[:16], dtype=torch.float)
temp_images = temp_images.view(-1, 3, 128, 128)

print(temp_images.shape)

save_image(temp_images, "images_end.png")

# Save the data as a numpy file
np.savez("font_data.npz", images=images)
