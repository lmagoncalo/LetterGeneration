import torch
from PIL import Image
from torch import nn
import lpips
from torchvision import transforms


class PerceptualLoss2(nn.Module):
    def __init__(self, target):
        super(PerceptualLoss2, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_fn = lpips.LPIPS(net='vgg')

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.target = Image.open(target)
        self.target = self.transform(self.target)
        self.target = self.target.unsqueeze(0)
        self.target = self.target.to(self.device)

    def forward(self, img):
        d = self.loss_fn(img, self.target)

        return d.squeeze()

    @classmethod
    def get_classname(cls):
        return "Perceptual 2 Loss"
