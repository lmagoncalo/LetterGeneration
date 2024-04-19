from pathlib import Path

import clip
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

from utils import wget_file


class AestheticLoss(nn.Module):
    def __init__(self, model=None, preprocess=None):
        super(AestheticLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = Path("saves/ava_vit_b_16_linear.pth")

        if not self.model_path.exists():
            wget_file(
                "https://cdn.discordapp.com/attachments/821173872111517696/921905064333967420/ava_vit_b_16_linear.pth",
                self.model_path)

        layer_weights = torch.load(self.model_path, map_location=self.device)
        self.ae_reg = nn.Linear(512, 1).to(self.device)

        self.ae_reg.bias.data = layer_weights["bias"].to(self.device)
        self.ae_reg.weight.data = layer_weights["weight"].to(self.device)

        self.clip_model = "ViT-B/16"

        if model is None:
            print(f"Loading CLIP model: {self.clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

    def forward(self, img):
        img = img.to(self.device)

        img = torchvision.transforms.functional.resize(img, (224, 224))

        image_features = self.model.encode_image(img)

        aes_rating = self.ae_reg(F.normalize(image_features.float(), dim=-1)).to(self.device)

        return aes_rating[0]

    @classmethod
    def get_classname(cls):
        return "Aesthetic Loss"
