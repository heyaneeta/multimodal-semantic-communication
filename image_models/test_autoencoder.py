import torch

from autoencoder import AutoEncoder

device = "cpu"

model = AutoEncoder()

model.load_state_dict(
    torch.load(
        "autoencoder.pth",
        map_location=device
    )
)

model.eval()

x = torch.randn(1,576)

with torch.no_grad():

    compressed,reconstructed = model(x)

mse = ((x-reconstructed)**2).mean()

print("MSE:",mse.item())