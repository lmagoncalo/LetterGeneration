import numpy as np
import pydiffvg
import torch
from torch import nn


class CircleRender(nn.Module):
    def __init__(self, canvas_size=224):
        super(CircleRender, self).__init__()

        self.canvas_size = canvas_size

        radius = 100
        segments = 30
        num_control_points = [2] * segments
        bias = (112, 112)
        points = []
        avg_degree = 360 / (segments * 3)
        for i in range(0, segments*3):
            point = (np.cos(np.deg2rad(i * avg_degree)), np.sin(np.deg2rad(i * avg_degree)))
            points.append(point)

        points = torch.tensor(points)
        points = (points) * radius + torch.tensor(bias).unsqueeze(dim=0)

        points = torch.FloatTensor(points)
        path = pydiffvg.Path(num_control_points=torch.LongTensor(num_control_points),  points=points, stroke_width=torch.tensor(0.0), is_closed=True)
        self.shapes = [path]

        polygon_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0, 0, 0, 1]), stroke_color=None)
        self.shape_groups = [polygon_group]

        path.points.requires_grad = True
        self.points_vars = [path.points]

        self.lrate = 1.0

    def get_optim(self):
        # Optimizers
        self.optim = torch.optim.Adam(self.points_vars, self.lrate)
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

