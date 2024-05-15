import torch
from torch import nn
from torchvision.models import mobilenet_v2
import torchvision.transforms.functional as TF

from utils import MakeCutouts


# {'"SANS_SERIF"': 0, '"SERIF"': 1, '"DISPLAY"': 2, '"HANDWRITING"': 3, '"MONOSPACE"': 4}

class StyleLoss(nn.Module):
    def __init__(self, model_path="saves/style_classifier.pth", num_cutouts=20):
        super(StyleLoss, self).__init__()
        """load the classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_target = 128

        self.style_target = torch.tensor([2]).repeat(num_cutouts).to(self.device)

        self.model = mobilenet_v2()  # Adjust num_classes as needed
        self.model.classifier[1] = torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=5)
        self.model.to(self.device)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.mk = MakeCutouts(cut_size=self.image_target, cutn=num_cutouts).to(self.device)

    def forward(self, img):
        img = img.to(self.device)

        # Resize image
        # img = TF.resize(img, (self.image_target, self.image_target))
        img = self.mk(img)

        result = self.model(img)

        loss = self.criterion(result, self.style_target)

        return loss

    @classmethod
    def get_classname(cls):
        return "Style Loss"