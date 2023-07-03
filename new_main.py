import os
import random
import string

import clip
import numpy as np
import pydiffvg
import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from losses import *
from render import SingleRender

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(device)


def rect_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return pts


def get_bezier_circle(_radius=1, segments=4, bias=None):
    new_points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments * 3)
    for i in range(0, segments * 3):
        point = (np.cos(np.deg2rad(i * avg_degree)), np.sin(np.deg2rad(i * avg_degree)))
        new_points.append(point)
    new_points = torch.tensor(new_points)
    new_points = new_points * _radius + torch.tensor(bias).unsqueeze(dim=0)
    new_points = new_points.type(torch.FloatTensor)
    return new_points


img_size = 224
iterations = 1001
seed = random.randint(0, 10000)

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

letters = list(string.ascii_uppercase)

renders = [SingleRender(canvas_size=100) for _ in range(26)]
optims = [render.get_optim() for render in renders]


model, preprocess = clip.load("ViT-B/32", device=device)

loss_functions = [CLIPLoss(f"The letter {letters[i]}", model=model, preprocess=preprocess) for i in range(26)]


for i in (pbar := tqdm(range(iterations))):
    for optim in optims:
        if i == int(iterations * 0.5):
            for g in optim.param_groups:
                g['lr'] /= 10
        if i == int(iterations * 0.75):
            for g in optim.param_groups:
                g['lr'] /= 10

    for l in range(26):
        img = renders[l].render()

        optims[l].zero_grad()
    
        loss = loss_functions[l](img)
    
        loss.backward()
    
        optims[l].step()

    if i % 50 == 0:
        newpath = f"results/{i}/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for render, letter in zip(renders, letters):
            render.save(path=f"results/{i}/{letter}.svg")
        # pydiffvg.imwrite(img.cpu(), f"results/{i}.png", gamma=1.0)

