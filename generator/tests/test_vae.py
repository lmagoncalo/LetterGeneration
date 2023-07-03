import torch

from generator.vae import VQVAE

vae = VQVAE()

i = torch.rand(1, 3, 224, 224)

out, latent_loss = vae(i)

print(out.shape, latent_loss)
