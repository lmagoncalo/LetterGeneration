"""
import pydiffvg
import torch

from losses import XingLoss

img_size = 200

num_control_points = torch.zeros(2, dtype=torch.int32) + 2

p0 = (10, 10)
p1 = (200, 120)
p2 = (10, 100)  # p2 = (10, 300) - 0.0005
p3 = (200, 10)
p4 = (10, 100)  # p4 = (200, 100) - 0.0004
p5 = (100, 200)
# loss - 0.0009

points = [p0, p1, p2, p3, p4, p5]
points = torch.tensor(points).float()

path = pydiffvg.Path(num_control_points=num_control_points,
                     points=points,
                     stroke_width=torch.tensor(1.0),
                     is_closed=False)
shapes = [path]

polygon_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None,
                                    stroke_color=torch.tensor([0, 0, 0, 1]))
shape_groups = [polygon_group]

# Just some diffvg setup
scene_args = pydiffvg.RenderFunction.serialize_scene(img_size, img_size, shapes, shape_groups)
render = pydiffvg.RenderFunction.apply

pydiffvg.save_svg(f"results/test.svg", img_size, img_size, shapes, shape_groups)

loss_function = XingLoss()

points_vars = []
for path in shapes:
    points_vars.append(path.points)

loss = loss_function(points_vars)

print(loss)
"""
import numpy as np
from vendi_score import vendi

samples = [1, 2, 3, 4, 5, 6, 7]
k = lambda a, b: np.exp(-np.abs(a - b))

print(vendi.score(samples, k))
