import numpy as np
import pydiffvg
import torch
from torch import nn


class Render(nn.Module):
    def __init__(self, letter, image_size=224):
        super(Render, self).__init__()

        self.image_size = image_size

        # Use float32 or else they translate somehow
        # Round the points or else it raises an error
        points = torch.FloatTensor(np.load(f"data/{letter}.npy"))
        points = torch.cat((points, points[-1].view(1, 2)), dim=0)
        points /= torch.max(points)
        points *= (image_size * 0.9)
        points += (image_size * 0.05)  # translate to center

        # TODO - Augment number of points


        # background shape
        p0 = [0, 0]
        p1 = [self.image_size, self.image_size]
        background = pydiffvg.Rect(p_min=torch.tensor(p0), p_max=torch.tensor(p1))

        num_control_points = torch.zeros(len(points) // 3, dtype=torch.int32) + 2

        polygon = pydiffvg.Path(num_control_points=num_control_points, points=points, is_closed=True)

        self.shapes = [background, polygon]

        background_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([0, 0, 0, 1]), stroke_color=None)
        polygon_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([1]), fill_color=torch.tensor([1, 1, 1, 1]), stroke_color=None)

        self.shape_groups = [background_group, polygon_group]

        polygon.points.requires_grad = True
        self.points_vars = [polygon.points]

    def get_optim(self):
        # Optimizers
        self.optim = torch.optim.Adam(self.points_vars, 1.0)
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

        pydiffvg.imwrite(img.cpu(), path, gamma=1.0)

