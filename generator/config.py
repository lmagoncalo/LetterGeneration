import string

import torch

IMG_SIZE = 64
BATCH_SIZE = 200
TIMESTEPS = 1000
# NUM_CLASSES = len(list(string.ascii_uppercase))
NUM_CLASSES = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
