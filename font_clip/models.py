import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, resnet50, ResNet50_Weights
from transformers import DistilBertModel, DistilBertConfig
import transformers


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, trainable=False, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # Adjust num_classes as needed
        else:
            self.model = resnet50()

        self.model.fc = Identity()


        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", trainable=False, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
            # self.model = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=256,
            dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class FontCLIPModel(nn.Module):
    def __init__(
        self,
        image_embedding=2048,
        text_embedding=768,
        trainable=True
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(trainable=trainable)
        self.text_encoder = TextEncoder(trainable=trainable)
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)

    def encode_image(self, image):
        x = self.image_encoder(image)
        return self.image_projection(x)

    def encode_text(self, text, attention_mask):
        x = self.text_encoder(input_ids=text, attention_mask=attention_mask)
        return self.text_projection(x)

    def forward(self, images, text, attention_mask):
        # Getting Image and Text Features
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids=text, attention_mask=attention_mask)

        # print("Image Features Shape: ", image_features.shape, "Text Features Shape: ", text_features.shape)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T)
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)

        return loss.mean()
