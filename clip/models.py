import torch
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, image_embedding):
        super().__init__()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights)  # Adjust num_classes as needed
        self.model.classifier[1] = torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=image_embedding)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)