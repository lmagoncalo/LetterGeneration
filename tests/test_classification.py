import torch
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image

from losses import ClassificationLoss
from utils import to_tensor

loss_function = ClassificationLoss("A")

transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Lambda(to_tensor),  # Use ToTensor if image is PIL Image
            transforms.RandomInvert(p=1.0),
        ])


img = Image.open("A.png").convert('RGB')
# img = transforms.ToTensor()(img).unsqueeze(0)
img = transform(img).unsqueeze(0)

out = loss_function(img).squeeze()

# _, pos = torch.max(out, dim=0)
# print(pos)
print(out)

img = Image.open("new_A.jpg").convert('RGB')
img = transform(img).unsqueeze(0)

out = loss_function(img).squeeze()

# _, pos = torch.max(out, dim=0)
# print(pos)
print(out)

img = Image.open("B.png").convert('RGB')
img = transform(img).unsqueeze(0)

out = loss_function(img).squeeze()

# _, pos = torch.max(out, dim=0)
# print(pos)
print(out)
