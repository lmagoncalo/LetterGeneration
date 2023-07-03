import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models


class PerceptualLoss(nn.Module):
    def __init__(self, target, pretrained=True, normalize=True, pre_relu=True):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(PerceptualLoss, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # VGG using perceptually-learned weights (LPIPS metric)
        self.normalize = normalize
        self.pretrained = pretrained

        self.feature_extractor = PerceptualLoss._FeatureExtractor(pretrained, pre_relu).to(self.device)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.resize = transforms.Compose([
            transforms.Resize((224, 224)),
        ])

        # Transform — Transform PIL image to Tensor and resize and normalize
        self.target = target.convert('RGB')
        self.target = transform(self.target)
        self.target = self.target.unsqueeze(0)
        self.target = self.target.to(self.device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, img):
        """Compare VGG features of two inputs."""

        img = self.resize(img)
        img = img.to(self.device)

        # Get VGG features
        pred = self.feature_extractor(img)
        target = self.feature_extractor(self.target)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [torch.sum((p - t) ** 2, 1) for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t) ** 2, 1) for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs).mean(0)

    @classmethod
    def get_classname(cls):
        return "Perceptual Loss"

    class _FeatureExtractor(nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(PerceptualLoss._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=pretrained).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats
