import random

import pydiffvg
import torch
from torch import nn


class SingleRender(nn.Module):
    def __init__(self, canvas_size=224):
        super(SingleRender, self).__init__()

        self.canvas_size = canvas_size

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
        points[:, 0] *= self.canvas_size
        points[:, 1] *= self.canvas_size
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(0.0),
                             is_closed=True)
        self.shapes = [path]

        polygon_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0, 0, 0, 1]),
                                            stroke_color=None)
        self.shape_groups = [polygon_group]

        path.points.requires_grad = True
        self.points_vars = [path.points]

    def get_optim(self):
        # Optimizers
        self.optim = torch.optim.Adam(self.points_vars, 1.0)
        return self.optim

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_size, self.canvas_size, self.shapes, self.shape_groups)
        img = render(self.canvas_size, self.canvas_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def save(self, path):
        pydiffvg.save_svg(path, self.canvas_size, self.canvas_size, self.shapes, self.shape_groups)

