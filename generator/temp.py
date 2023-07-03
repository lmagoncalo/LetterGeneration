import torch
from torch import nn


third_pos_loss_fn = nn.CrossEntropyLoss()

pen_state = torch.rand(32 * 96, 2).float()
target_pen_state = torch.rand(32 * 96).long()
loss = third_pos_loss_fn(pen_state.cpu(), target_pen_state.cpu())
print(loss.item())
