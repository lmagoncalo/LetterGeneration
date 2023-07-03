import torch
import torch.nn.functional as F
from torch import nn

from config import *


class Diffusion(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model.to(DEVICE)

        # Define beta schedule
        self.timesteps = TIMESTEPS
        self.betas = self.linear_beta_schedule(timesteps=self.timesteps)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(DEVICE)

    def forward_diffusion_sample(self, x_0, t):
        """
        Takes an image and a timestep as input and returns the noisy version of it
        """
        noise = torch.randn_like(x_0).float()
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(DEVICE) * x_0.to(DEVICE) + sqrt_one_minus_alphas_cumprod_t.to(
            DEVICE) * noise.to(DEVICE), noise.to(DEVICE)

    @torch.no_grad()
    def sample_timestep(self, x_0, t, label=None):
        """
        Calls the model to predict the noise in the image and returns the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        x = x_0[:, :, :2]

        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        pred_noise, pen_state = self.model(x, t, label)
        # pred_noise = self.model(x, t, label)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return model_mean, pen_state
            # return model_mean
        else:
            noise = torch.randn_like(pred_noise)
            return model_mean + torch.sqrt(posterior_variance_t) * noise, pen_state
            # return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_plot_image(self):
        # Sample noise
        img = torch.randn((1, 96, 2), device=DEVICE)
        # pen_state = torch.zeros((1, 96, 1), device=DEVICE)
        label = torch.randint(0, NUM_CLASSES, size=(1,), device=DEVICE)

        for i in range(0, self.timesteps)[::-1]:
            t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
            img, pen_state = self.sample_timestep(img, t, label=label)

        pen_state = pen_state.reshape(-1, 2)
        pen_state = torch.argmax(pen_state, dim=1)
        pen_state = pen_state.reshape(img.size(0), img.size(1), -1)
        img = torch.concat((img, pen_state), dim=2)

        return img

    def loss(self, x_0, label, t):
        x = x_0[:, :, :2]
        target_pen_state = x_0[:, :, 2]

        x_noisy, noise = self.forward_diffusion_sample(x, t)

        noise_pred, pen_state = self.model(x_noisy, t, y=label)

        target_pen_state = target_pen_state.reshape(x_0.size(0) * x_0.size(1), ).long()

        pen_state = pen_state.reshape(x_0.size(0) * x_0.size(1), -1).float()

        l1_loss = F.mse_loss(noise, noise_pred)

        # Multiply each element loss by the alpha at timestep t
        pen_loss = F.cross_entropy(pen_state, target_pen_state, reduction="none")
        pen_loss = pen_loss.reshape(x_0.size(0), x_0.size(1), -1)
        pen_loss = torch.mean(pen_loss, dim=1).squeeze(0)

        alpha_value = self.alphas_cumprod_prev.gather(-1, t.cpu()).to(DEVICE)

        pen_loss = torch.mean(alpha_value * pen_loss)

        loss = l1_loss + pen_loss

        losses = {"l1_loss": l1_loss, "pen_loss": pen_loss}

        return loss, losses
