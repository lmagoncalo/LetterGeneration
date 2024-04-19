import torch
from PIL import Image
from torchvision import transforms

from losses import NoveltyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_function = NoveltyLoss()

img = Image.open("A.png").convert('RGB')
img = transforms.ToTensor()(img).unsqueeze(0).to(device)

print(img)

img_2 = Image.open("new_A.jpg").convert('RGB')
img_2 = transforms.ToTensor()(img_2).unsqueeze(0).to(device)

out = loss_function(img)
out_2 = loss_function(img_2)

print(out, out_2)




