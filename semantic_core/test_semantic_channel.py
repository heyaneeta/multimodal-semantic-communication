import numpy as np
import torch

from autoencoder_model import SemanticAutoencoder

FEATURE_PATH = "semantic_core/features.npy"
MODEL_PATH = "semantic_core/semantic_autoencoder.pth"

features = np.load(FEATURE_PATH)

original = features[0]

model = SemanticAutoencoder(input_dim=576, latent_dim=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def add_awgn(x, noise_level=0.1):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

def packet_loss(x, loss_rate=0.2):
    mask = np.random.rand(*x.shape) > loss_rate
    return x * mask

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    mse_value = mse(a, b)

    if mse_value == 0:
        return float("inf")

    max_value = np.max(a)

    return 20 * np.log10(max_value / np.sqrt(mse_value))

with torch.no_grad():
    original_tensor = torch.tensor(original, dtype=torch.float32).unsqueeze(0)

    compressed = model.encode(original_tensor).numpy()[0]

    noisy = add_awgn(compressed, noise_level=0.1)

    corrupted = packet_loss(noisy, loss_rate=0.2)

    corrupted_tensor = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0)

    recovered = model.decode(corrupted_tensor).numpy()[0]

print("Original feature shape:", original.shape)
print("Compressed semantic shape:", compressed.shape)

print("\nOriginal first 10:")
print(original[:10])

print("\nCompressed first 10:")
print(compressed[:10])

print("\nCorrupted compressed first 10:")
print(corrupted[:10])

print("\nRecovered first 10:")
print(recovered[:10])

print("\nMSE:", mse(original, recovered))
print("PSNR:", psnr(original, recovered))