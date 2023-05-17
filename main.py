import random

import numpy as np
import pydiffvg
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from losses import *
from losses.perceptual_loss import PerceptualLoss

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
num_iter = 1001
seed = random.randint(0, 10000)

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

"""
num_segments = 10
radius = 0.05
num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
points = []
p0 = (random.random(), random.random())
points.append(p0)
for j in range(num_segments):
    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
    p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
    p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
    points.append(p1)
    points.append(p2)
    if j < num_segments - 1:
        points.append(p3)
        p0 = p3

points = torch.tensor(points)
points[:, 0] *= img_size
points[:, 1] *= img_size
path = pydiffvg.Path(num_control_points=num_control_points,
                     points=points,
                     stroke_width=torch.tensor(0.0),
                     is_closed=True)
shapes = [path]

polygon_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0, 0, 0, 1]),
                                    stroke_color=None)
shape_groups = [polygon_group]

path.points.requires_grad = True
points_vars = [path.points]
"""
num_paths = 100
shapes = []
shape_groups = []
for i in range(num_paths):
    num_segments = random.randint(3, 5)
    num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
    points = []
    p0 = (random.random(), random.random())
    points.append(p0)
    for j in range(num_segments):
        radius = 0.05
        p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
        p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
        p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
        points.append(p1)
        points.append(p2)
        if j < num_segments - 1:
            points.append(p3)
            p0 = p3
    points = torch.tensor(points)
    points[:, 0] *= img_size
    points[:, 1] *= img_size
    path = pydiffvg.Path(num_control_points=num_control_points,
                         points=points,
                         stroke_width=torch.tensor(1.0),
                         is_closed=True)
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                     fill_color=torch.tensor([random.random(),
                                                              random.random(),
                                                              random.random(),
                                                              random.random()]))
    shape_groups.append(path_group)

points_vars = []
for path in shapes:
    path.points.requires_grad = True
    points_vars.append(path.points)

# Just some diffvg setup
scene_args = pydiffvg.RenderFunction.serialize_scene(img_size, img_size, shapes, shape_groups)
render = pydiffvg.RenderFunction.apply

optims = [torch.optim.Adam(points_vars, lr=1.0)]

# loss_function = CLIPSimilarityLoss("dogcat.jpg")
# loss_function = ClassificationLoss(letter=0)
# loss_function = PerceptualLoss2("dogcat.jpg")
# loss_function = PerceptualLoss("dogcat.jpg")
# loss_function = XingLoss()
# loss_function = VendiLoss()

for i in (pbar := tqdm(range(num_iter))):
    """
    if i == int(num_iter * 0.5):
        for g in points_optim.param_groups:
            g['lr'] /= 10
    if i == int(num_iter * 0.75):
        for g in points_optim.param_groups:
            g['lr'] /= 10
    """
    img = render(img_size, img_size, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                      device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

    for optim in optims:
        optim.zero_grad()

    loss = loss_function(img)
    loss.backward()

    for optim in optims:
        optim.step()

    if i % 100 == 0:
        pydiffvg.save_svg(f"results/{i}.svg", img_size, img_size, shapes, shape_groups)
        # pydiffvg.imwrite(img.cpu(), f"results/{i}.png", gamma=1.0)

    pbar.set_description(f"Loss {loss.item()}")
