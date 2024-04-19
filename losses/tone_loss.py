import math
import numpy as np
import torchvision
from torch import nn


class ToneLoss(nn.Module):
    def __init__(self):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = 2
        self.im_init = None
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(75, 75), sigma=(10))

        self.image_target = 224

    def set_image_init(self, im_init):
        self.im_init = im_init
        self.init_blurred = self.blurrer(self.im_init)

        torchvision.utils.save_image(self.init_blurred, 'init_blurred.png')

    """
    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5) * ((step-300) / 20) ** 2)
        else:
            return self.dist_loss_weight
    """

    def get_scheduler(self, step=None, end_step=1000, start_value=0.001):
        if step is not None:
            return start_value + (self.dist_loss_weight - start_value) * step / end_step
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)

    @classmethod
    def get_classname(cls):
        return "Tone Loss"