import os
from statistics import mean

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from config import *
from vae import VQVAE


class ImageDataset(Dataset):
    def __init__(self, base_path, transform):
        self.base_path = base_path

        # list files in img directory
        files = os.listdir(self.base_path)
        self.image_paths = []

        for file in files:
            # make sure file is an image
            if file.endswith(('.jpg', '.png', 'jpeg')):
                img_path = path + file
                self.image_paths.append(img_path)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)

        image = self.transform(image)

        return image


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

path = "images/"

dataset = ImageDataset(path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64)

model = VQVAE().to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=3e-4)

criterion = nn.MSELoss()

latent_loss_weight = 0.25
sample_size = 10
n_epochs = 21

for epoch in tqdm(range(n_epochs + 1)):
    for g in optimizer.param_groups:
        g['lr'] *= 0.97

    recon_losses = []
    latent_losses = []
    for i, img in enumerate(dataloader):
        model.zero_grad()

        img = img.to(DEVICE)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()

        recon_losses.append(recon_loss.item())
        latent_losses.append(latent_loss.item())

    print(f"Epoch: {epoch} | Recon Loss: {mean(recon_losses)} | Latent Loss: {mean(latent_losses)}")

    if epoch % 5 == 0:
        model.eval()
        sample = img[:sample_size]

        with torch.no_grad():
            out, _ = model(sample)

        save_image(
            torch.cat([sample, out], 0),
            f"results/result_{str(epoch)}.png",
            nrow=sample_size,
            normalize=True,
            range=(-1, 1),
        )

torch.save(model.state_dict(), "vqvae.pth")
