import random

import numpy as np
import pydiffvg
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image


def rect_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return pts


class Render(nn.Module):
    def __init__(self, letter, image_size=224):
        super(Render, self).__init__()

        self.image_size = image_size

        canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(f"data/{letter}.svg")

        self.shapes = shapes_init
        self.shape_groups = shape_groups_init

        max_x = 0
        max_y = 0
        for path in shapes_init:
            if max_x < path.points[:, 0].max():
                max_x = path.points[:, 0].max()
            if max_y < path.points[:, 1].max():
                max_y = path.points[:, 1].max()

        self.points_vars = []
        for path in shapes_init:
            path.points[:, 0] /= max_x
            path.points[:, 1] /= max_y

            path.points *= (self.image_size * 0.7)

            path.points += (self.image_size * 0.10)

            path.points.requires_grad = True
            self.points_vars.append(path.points)

    def get_optim(self):
        # Optimizers
        self.optim = torch.optim.Adam(self.points_vars, 0.5)
        return self.optim

    def render(self, sizes=None):
        """
        if sizes is None:
            sizes = [32, 64, 224]

        render = pydiffvg.RenderFunction.apply

        images = {}
        for size in sizes:
            print("Rendering", size)

            # Define the canvas size
            canvas_size = size

            # Serialize the scene
            scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, self.shapes, self.shape_groups)

            # Render the scene
            img = render(canvas_size, canvas_size, 2, 2, 0, None, *scene_args)

            # Post-process the image
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                              device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
            img = img[:, :, :3]
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

            images[size] = img

        return images
        """

        canvas_size = self.image_size
        render = pydiffvg.RenderFunction.apply

        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, self.shapes, self.shape_groups)

        # Render the scene
        img = render(canvas_size, canvas_size, 2, 2, 0, None, *scene_args)

        # Post-process the image
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        return img

    def save_svg(self, path, sizes=None):
        """
        if sizes is None:
            sizes = [32, 64, 224]

        for size in sizes:
            # Define the canvas size
            canvas_size = size

            # Save the SVG file
            pydiffvg.save_svg(f"{path}_{size}x{size}.svg", width=canvas_size, height=canvas_size, shapes=self.shapes, shape_groups=self.shape_groups)
        """
        canvas_size = self.image_size
        pydiffvg.save_svg(path, width=canvas_size, height=canvas_size, shapes=self.shapes, shape_groups=self.shape_groups)

    def save_png(self, path):
        canvas_size = self.image_size

        render = pydiffvg.RenderFunction.apply

        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, self.shapes, self.shape_groups)

        # Render the scene
        img = render(canvas_size, canvas_size, 2, 2, 0, None, *scene_args)

        # Post-process the image
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        save_image(img, path)


