from statistics import mean

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from data import FontDataset, SketchDataset
from diffusion import Diffusion
from simpleunet import UNetModel
from utils import draw_three

# dataset = FontDataset(fonts_path="fonts.npz")
dataset = SketchDataset(image_paths="datasets", category=["spider.npz"])

epochs = 201
lr = 1e-4

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

# Model_channel = number of points
model = UNetModel(in_channels=3, model_channels=96, out_channels=3, num_res_blocks=2, attention_resolutions=(16, 8),
                  dropout=0.0, channel_mult=(1, 2, 3, 4), num_classes=NUM_CLASSES, use_checkpoint=False, num_heads=4,
                  num_heads_upsample=-1, use_scale_shift_norm=True).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=lr)

diffusion = Diffusion(model=model).to(DEVICE)

for epoch in range(epochs):
    for g in optimizer.param_groups:
        g['lr'] *= 0.97

    l1_losses = []
    pen_losses = []
    for batch, label in tqdm(dataloader):
        batch = batch.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()

        timesteps = torch.randint(0, TIMESTEPS, (batch.shape[0],)).long().to(DEVICE)

        loss, losses = diffusion.loss(batch, label, timesteps)

        loss.backward()

        optimizer.step()

        l1_losses.append(losses["l1_loss"].item())
        pen_losses.append(losses["pen_loss"].item())

    print(f"Epoch: {epoch} | L1 Loss: {mean(l1_losses)} | Pen Loss: {mean(pen_losses)}")

    if epoch % 5 == 0:
        gen_batch = diffusion.sample_plot_image()
        temp = gen_batch[0]
        img = draw_three(temp)
        img.save(f"results/image_{epoch}.png")

