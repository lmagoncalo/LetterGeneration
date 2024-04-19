import clip
import torch
import math
import torchvision.transforms as T
from torch import nn
from torch.nn import functional as F

from utils import MakeCutouts


class CLIPLoss(nn.Module):
    def __init__(self, prompts, num_cutouts, model=None, preprocess=None, clip_model="ViT-B/32"):
        super(CLIPLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mk = MakeCutouts(cut_size=224, cutn=num_cutouts).to(self.device)

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
            text_features = self.model.encode_text(text_inputs)

            self.embed_normed = F.normalize(text_features.unsqueeze(0), dim=2)

    def forward(self, img):
        img = img.to(self.device)

        into = self.mk(img)

        image_features = self.model.encode_image(into)

        input_normed = F.normalize(image_features.unsqueeze(1), dim=2)

        distance = input_normed.sub(self.embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)

        return distance.mean()

        # cosine_similarity = torch.cosine_similarity(self.text_features, image_features, dim=-1).mean()
        # return cosine_similarity

    @classmethod
    def get_classname(cls):
        return "CLIP Loss"