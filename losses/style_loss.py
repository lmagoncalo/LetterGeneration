import torch
from torch import nn
from torchvision.models import mobilenet_v2

# {'"SANS_SERIF"': 0, '"SERIF"': 1, '"DISPLAY"': 2, '"HANDWRITING"': 3, '"MONOSPACE"': 4}

class StyleLoss(nn.Module):
    def __init__(self, model_path="saves/style_classifier.pth"):
        super(StyleLoss, self).__init__()
        """load the classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_target = 64

        self.style_target = torch.tensor([2]).to(self.device)

        self.model = mobilenet_v2()  # Adjust num_classes as needed
        self.model.classifier[1] = torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=10)
        self.model.to(self.device)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # self.target_rating = torch.ones(size=(num_cutouts, 1)) * -1
        # self.target_rating = self.target_rating.to(self.device)

    def forward(self, img):
        img = img.to(self.device)

        result = self.model(img)

        loss = self.criterion(result, self.style_target)

        # print(loss.item())

        # aes_loss = (nll_value - self.target_rating).square().mean()

        return loss

    @classmethod
    def get_classname(cls):
        return "Style Loss"