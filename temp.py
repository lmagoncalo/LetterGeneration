import torch

images = torch.randn(8, 3, 224, 224)
input_ids = torch.randint(5, 300, size=(8, 25))
attention_mask = torch.ones(8, 25)

print(images.shape, input_ids.shape, attention_mask.shape)