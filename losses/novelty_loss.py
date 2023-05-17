import torch
from torch import nn

from losses.vae import VAE


class NoveltyLoss(nn.Module):
    def __init__(self, model_path="saves/pretrained_autoencoder.pth"):
        super(NoveltyLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VAE().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.criterion = nn.L1Loss().to(self.device)

    def forward(self, imgs):
        imgs_recon = self.model(imgs)

        recon_loss = self.criterion(imgs_recon, imgs)

        return recon_loss

    @classmethod
    def get_classname(cls):
        return "Novelty Loss"
