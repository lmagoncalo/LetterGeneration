import torch
from torch import nn
from torchvision import transforms

class Autoencoder(nn.Module):
    def __init__(self, image_size=128, latent_dim=100):
        super(Autoencoder, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (image_size // 16) * (image_size // 16), latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * (image_size // 16) * (image_size // 16)),
            nn.Unflatten(1, (256, (image_size // 16), (image_size // 16))),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class NoveltyLoss(nn.Module):
    def __init__(self, model_path="saves/conv_autoencoder.pth"):
        super(NoveltyLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Autoencoder().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.criterion = nn.MSELoss().to(self.device)

    def forward(self, imgs):
        imgs_recon = self.model(imgs)

        recon_loss = self.criterion(imgs_recon, imgs)

        # We want to maximize the reconstruction loss, so we know it is a new sample
        return 1/recon_loss

    @classmethod
    def get_classname(cls):
        return "Novelty Loss"
