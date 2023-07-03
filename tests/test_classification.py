import torch
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image

from losses import ClassificationLoss

loss_function = ClassificationLoss()


img = Image.open("A.png").convert('RGB')
img = transforms.ToTensor()(img).unsqueeze(0)

out = loss_function(img).squeeze()

# _, pos = torch.max(out, dim=0)
# print(pos)
print(out)

img = Image.open("new_A.jpg").convert('RGB')
img = transforms.ToTensor()(img).unsqueeze(0)

out = loss_function(img).squeeze()

# _, pos = torch.max(out, dim=0)
# print(pos)
print(out)

img = Image.open("B.png").convert('RGB')
img = transforms.ToTensor()(img).unsqueeze(0)

out = loss_function(img).squeeze()

# _, pos = torch.max(out, dim=0)
# print(pos)
print(out)

# /home/cdv/miniconda/envs/diffusion/bin/python