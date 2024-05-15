import torch
from torch import nn
from torch.nn import functional as F
from transformers import DistilBertTokenizer

from font_clip.models import FontCLIPModel


class FontCLIPLoss(nn.Module):
    def __init__(self, prompts):
        super(FontCLIPLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FontCLIPModel().to(self.device)
        self.model.load_state_dict(torch.load("font_clip/font_clip_model.pth"))

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        encoded_input = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=32, return_tensors='np')

        input_ids = torch.tensor(encoded_input['input_ids']).to(self.device)
        attention_mask = torch.tensor(encoded_input['attention_mask']).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(input_ids, attention_mask)

            self.embed_normed = F.normalize(text_features, dim=1)

    def forward(self, img):
        img = img.to(self.device)

        image_features = self.model.encode_image(img)

        input_normed = F.normalize(image_features, dim=1)

        distance = input_normed.sub(self.embed_normed).norm(dim=1).div(2).arcsin().pow(2).mul(2)
        return distance.mean()

        # cosine_similarity = torch.cosine_similarity(self.text_features, input_normed, dim=-1).mean()
        # return cosine_similarity

    @classmethod
    def get_classname(cls):
        return "Font CLIP Loss"