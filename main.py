import random

import font_clip
import numpy as np
import pydiffvg
import torch
from tqdm import tqdm

from losses import *
from render import Render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)

img_size = 128
num_iter = 1001
seed = random.randint(0, 10000)
prompt = "a lion"
# letters = list(string.ascii_uppercase)
letter = "R"
num_cutouts = 50

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# model, preprocess = font_clip.load("ViT-B/32", device=device)

# render = Render(canvas_size=img_size, letter="C")
# render = CircleRender(canvas_size=img_size)
# optim = render.get_optim()

render = Render(letter=letter, image_size=img_size)
optim = render.get_optim()

# losses = [CLIPLoss(prompt)]
# loss_function = CLIPLoss(prompt, model=model, preprocess=preprocess)
# loss_function = SSIMLoss(Image.open("B.png"))
# losses = [ClassificationLoss(letter="Z")]
# losses = [ClassificationLoss(letter="D", num_cutouts=num_cutouts)]
# loss_function = PerceptualLoss2("dogcat.jpg")
# loss_function = PerceptualLoss("dogcat.jpg")
# loss_function = XingLoss()
# loss_function = VendiLoss()

# tone_loss = ToneLoss()
# clip_loss = CLIPLoss(prompt, num_cutouts=num_cutouts)
# conformal_loss = ConformalLoss(parameters=render.points_vars, shape_groups=render.shape_groups, letter=letter)
# style_loss = StyleLoss()
novelty_loss = NoveltyLoss()

print("Starting generating...")
for i in range(num_iter):
    if i == int(num_iter * 0.5) or i == int(num_iter * 0.75):
        for g in optim.param_groups:
            g['lr'] /= 10

    optim.zero_grad()

    image = render.render()

    # if i == 0:
    #     print("Tone Loss init image init")
    #     tone_loss.set_image_init(image)

    # loss = clip_loss(image)
    # loss_style = style_loss(image)
    # tone_loss_res = tone_loss(image, i)
    # loss_angles = conformal_loss()
    # loss = loss_style + (loss_angles * 0.5) + tone_loss_res
    # loss = style_loss(image)

    loss = novelty_loss(image) * 0.1

    print(loss.item())

    loss.backward()

    optim.step()

    if i % 100 == 0:
        # render.save_svg(f"results/letter_{i}_{letter}.svg")
        render.save_png(f"results/letter_{i}_{letter}.png")
        print(loss.item())

    # break
