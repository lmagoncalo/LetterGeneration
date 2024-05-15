import csv
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import DistilBertTokenizer

# {'"SANS_SERIF"': 0, '"SERIF"': 1, '"DISPLAY"': 2, '"HANDWRITING"': 3, '"MONOSPACE"': 4}

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define the root directory where your font dataset is stored
root_dir = "../classifier/dataset"

# Create an empty list to store images and labels
data = []

# Map category names to integer class labels
class_label_mapping = {}

def merge_entries(csv_file):
    merged_data = {}
    with open(csv_file, 'r', newline='\n') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            key = row[0].lower().replace(" ", "")
            words_to_add = row[1].split('/')  # Split the second column by backslash
            if key in merged_data:
                for word in words_to_add:
                    if word not in merged_data[key] and word:
                        merged_data[key].append(word)
            else:
                merged_data[key] = [word for word in words_to_add if word]

    return merged_data

merged_data = merge_entries("families.csv")
print(merged_data)

# Find maximum array length
max_len = max([len(merged_data[category]) for category in merged_data])

print("Maximum array length: ", max_len)


# Iterate over folders
for category_folder in tqdm(os.listdir(root_dir)):
    category_dir = os.path.join(root_dir, category_folder)
    metadata_path = os.path.join(category_dir, 'METADATA.pb')
    if os.path.exists(metadata_path) and category_folder in merged_data:
        # print(category_folder, merged_data[category_folder], ' '.join(merged_data[category_folder]))

        # Get style from previous read csv
        style = ' '.join(merged_data[category_folder])

        # Find font files in the category folder
        font_files = [file for file in os.listdir(category_dir) if file.endswith('.ttf') or file.endswith('.otf')]
        # print("Number of font files: ", len(font_files))
        # Generate images for each font file and letter
        for font_file in font_files:
            font_path = os.path.join(category_dir, font_file)
            try:
                font = ImageFont.truetype(font_path, size=60)
                for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    img = Image.new('RGB', (64, 64), color='white')
                    draw = ImageDraw.Draw(img)
                    left, top, right, bottom = draw.textbbox((0, 0), letter, font=font, align="center")
                    w = right - left
                    h = bottom - top
                    y = -top + (32 - (h // 2))
                    x = -left + (32 - (w // 2))
                    # x = (64 - (right - left)) // 2
                    # y = (37 - (bottom - top)) // 2
                    draw.text((x, y), letter, fill='black', font=font)
                    img_tensor = TF.pil_to_tensor(img) / 255
                    img_np = img_tensor.numpy()

                    letter_style = f"Letter {letter}, {style}"

                    # Tokenize letter style
                    encoded_input = tokenizer(letter_style, padding='max_length', truncation=True, max_length=32, return_tensors='np')

                    data.append((img_np, encoded_input['input_ids'], encoded_input['attention_mask']))
            except Exception as e:
                print(f"Error processing {font_path}: {e}")

# Shuffle the data
np.random.shuffle(data)

# Split data into images and labels
images = np.array([d[0] for d in data])
input_ids = np.array([d[1] for d in data])
attention_masks = np.array([d[2] for d in data])

print(f"Number of images: {len(images)}, Shape: {images.shape}")
print(f"Number of Input IDs: {len(input_ids)}, Shape: {input_ids.shape}")
print(f"Number of Attentions Maks: {len(attention_masks)}, Shape: {attention_masks.shape}")

temp_images = torch.tensor(images[:16], dtype=torch.float)
temp_images = temp_images.view(-1, 3, 64, 64)

print(temp_images.shape)

save_image(temp_images, "images_end.png")

# Save the data as a numpy file
np.savez("font_clip_data.npz", images=images, input_ids=input_ids, attention_masks=attention_masks)
