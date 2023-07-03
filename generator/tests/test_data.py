import os

import numpy as np
from torch.utils.data import DataLoader

from generator.data import FontDataset
from generator.utils import draw_three


# dataset = np.load("datasets/moon.npz", encoding='latin1', allow_pickle=True)
# dataset = np.load("fonts.npz", encoding='latin1', allow_pickle=True)

# print(dataset["data"].shape)

"""
dataset = FontDataset(fonts_path="fonts.npz")

batch_size = 2

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

sketch, labels = next(iter(dataloader))

print(sketch.shape, labels.shape)

img = draw_three(sketch[0], random_color=True)

img.save("tests/test_sketch.png")
"""

dataset = np.load("images.npz", encoding='latin1', allow_pickle=True)
images = dataset["images"]
print(images.shape)
