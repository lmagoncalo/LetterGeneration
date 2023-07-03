import clip
import torch
import math
import torchvision.transforms as T
from torch import nn
from torch.nn import functional as F


def range_loss(_input):
    return (_input - _input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(_input, size, align_corners=True):
    n, c, h, w = _input.shape
    dh, dw = size

    _input = _input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(_input.device, _input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        _input = F.pad(_input, (0, 0, pad_h, pad_h), 'reflect')
        _input = F.conv2d(_input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(_input.device, _input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        _input = F.pad(_input, (pad_w, pad_w, 0, 0), 'reflect')
        _input = F.conv2d(_input, kernel_w[None, None, None, :])

    _input = _input.reshape([n, c, h, w])
    return F.interpolate(_input, size, mode='bicubic', align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, _input):
        _input = T.Pad(_input.shape[2]//4, fill=0)(_input)
        side_y, side_x = _input.shape[2:4]
        max_size = min(side_x, side_y)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = _input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offset_x = torch.randint(0, abs(side_x - size + 1), ())
                offset_y = torch.randint(0, abs(side_y - size + 1), ())
                cutout = _input[:, :, offset_y:offset_y + size, offset_x:offset_x + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            # print(cutout.shape)
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class CLIPLoss(nn.Module):
    def __init__(self, prompts, model=None, preprocess=None, clip_model="ViT-B/32"):
        super(CLIPLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model = clip_model

        if model is None:
            print(f"Loading CLIP model: {self.clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

        text_inputs = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)

        self.mk = MakeCutouts(cut_size=224, cutn=10).to(self.device)

    def forward(self, img):
        img = img.to(self.device)

        into = self.mk(img)

        image_features = self.model.encode_image(into)

        input_normed = F.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.text_features.unsqueeze(0), dim=2)

        distance = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)

        return distance.mean()

        # cosine_similarity = torch.cosine_similarity(self.text_features, image_features, dim=-1).mean()
        # return cosine_similarity

    @classmethod
    def get_classname(cls):
        return "CLIP Loss"
