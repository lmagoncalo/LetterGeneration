import random

import pydiffvg
import torch
from torch import nn


class GridRender(nn.Module):
    def __init__(self, canvas_size=224):
        super(GridRender, self).__init__()

        self.canvas_size = canvas_size

        shapes = []
        shape_groups = []
        colors = []
        cell_size = int(self.canvas_size / 28)
        for r in range(28):
            cur_y = r * cell_size
            for c in range(28):
                cur_x = c * cell_size
                p0 = [cur_x, cur_y]
                p1 = [cur_x + cell_size, cur_y + cell_size]

                cell_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])
                colors.append(cell_color)

                path = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]), stroke_color=None,
                                                 fill_color=cell_color)
                shape_groups.append(path_group)

        color_vars = []
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)

        self.color_vars = color_vars

    def get_optim(self):
        # Optimizers
        self.optims = [torch.optim.Adam(self.color_vars, lr=0.1)]
        return self.optim

    def render(self):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_size, self.canvas_size, self.shapes, self.shape_groups)
        img = render(self.canvas_size, self.canvas_size, 2, 2, 0, None, *scene_args)

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Clip values
        # print(color_vars)
        for group in self.shape_groups:
            temp_color = torch.mean(group.fill_color.data[:-1])
            # temp_color = torch.round(temp_color)
            temp_color = temp_color.repeat(3)
            group.fill_color.data = torch.cat((temp_color, torch.ones(1, requires_grad=True)))

        return img

    def save(self, path):
        pydiffvg.save_svg(path, self.canvas_size, self.canvas_size, self.shapes, self.shape_groups)
