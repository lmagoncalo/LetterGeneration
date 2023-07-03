import torch
from torch import nn
from torchvision import transforms
from piq import SSIMLoss as SSIM


class SSIMLoss(nn.Module):
    def __init__(self, target):
        super(SSIMLoss, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.resize = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        # Transform — Transform PIL image to Tensor and resize and normalize
        self.target = target.convert('RGB')
        self.target = transform(self.target)
        self.target = self.target.unsqueeze(0)
        self.target = self.target.to(self.device)

        self.ssim_loss = SSIM()

    def forward(self, img):
        img = self.resize(img)
        img = img.to(self.device)

        img = torch.clip(img, min=0.0, max=1.0)

        loss = self.ssim_loss(img, self.target)

        return loss
