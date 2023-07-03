from generator.data import SketchDataset
from generator.simpleunet import UNetModel
from generator.config import *

dataset = SketchDataset(image_paths="datasets", category=["moon.npz", "airplane.npz", "fish.npz", "umbrella.npz", "train.npz", "spider.npz", "shoe.npz", "apple.npz", "lion.npz", "bus.npz"])

batch_size = 6

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

batch, out_dict = next(iter(dataloader))

batch = batch[:, :, :2]

# i = torch.rand(6, 96, 2)
print(batch.shape)

model = UNetModel(in_channels=96, model_channels=96, out_channels=3, num_res_blocks=2, attention_resolutions=(16, 8), dropout=0.0, channel_mult=(1, 2, 3, 4), num_classes=None,  use_checkpoint=False, num_heads=4, num_heads_upsample=-1, use_scale_shift_norm=True)

t = torch.full((6,), 0, dtype=torch.long)

model_output, pen_state = model(batch, t)

print(model_output.shape, pen_state.shape)

