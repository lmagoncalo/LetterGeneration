import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import VectorQuantizedVAE, Autoencoder

# Define the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Initialize the autoencoder
# model = VectorQuantizedVAE(3, 64, 128).to(device)
model = Autoencoder().to(device)
model.load_state_dict(torch.load("conv_autoencoder.pth"))
model.eval()

for image_path in ["example_0.png", "example_1.png", "example_2.png"]:
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    # Check if the image only has 1 channel
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)

    # save_image(img, "test.png")

    # print(img.shape, torch.max(img), torch.min(img))

    with torch.no_grad():
        x_recons = model(img)
        loss_recons = F.mse_loss(x_recons, img)
        print("Image:", image_path, "Loss:", loss_recons.item())

# Image: example_0.png Loss: 0.00070495082763955
# Image: example_1.png Loss: 0.000661723199300468
# Image: example_2.png Loss: 0.0013895293232053518
