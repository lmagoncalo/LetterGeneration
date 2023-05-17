import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights


# VS(K) = exp(tr(K/n @ log(K/n))) = exp(-sum_i lambda_i log lambda_i),


def get_inception(pretrained=True, pool=True):
    if pretrained:
        weights = Inception_V3_Weights.DEFAULT
    else:
        weights = None
    model = inception_v3(
        weights=weights, transform_input=True
    ).eval()
    if pool:
        model.fc = nn.Identity()
    return model


def inception_transforms():
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        ]
    )


def get_embeddings(
    images,
    model=None,
    transform=None,
    device=torch.device("cpu"),
):
    if type(device) == str:
        device = torch.device(device)
    if model is None:
        model = get_inception(pretrained=True, pool=True).to(device)
        transform = inception_transforms()
    if transform is None:
        transform = transforms.ToTensor()

    images = transform(images).to(device)
    output = model(images)

    return output


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * torch.log(p_)).sum()
    if q == "inf":
        return -torch.log(torch.max(p))
    return torch.log((p_ ** q).sum()) / (1 - q)


def score_dual(X, q=1, normalize=True):
    if normalize:
        # X = preprocessing.normalize(X, axis=1)
        X = F.normalize(X, dim=1)

    n = X.shape[0]
    S = X.T @ X
    # w = scipy.linalg.eigvalsh(S / n)
    w = torch.linalg.eigvalsh(S / n)
    # m = w > 0
    return torch.exp(entropy_q(w, q=q))


# TODO - Not working
class VendiLoss(nn.Module):
    def __init__(self):
        super(VendiLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, imgs):
        imgs = [img.squeeze() for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        X = get_embeddings(imgs)
        # n, d = X.shape
        # if n < d:
        #     return score_X(X)
        return score_dual(X)

    @classmethod
    def get_classname(cls):
        return "Vendi Loss"
