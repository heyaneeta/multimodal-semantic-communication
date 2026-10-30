import torch

checkpoint = torch.load("autoencoder.pth", map_location="cpu")

print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys:")
    for k in checkpoint.keys():
        print(k)