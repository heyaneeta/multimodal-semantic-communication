import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from autoencoder_model import SemanticAutoencoder

FEATURE_PATH = "semantic_core/features.npy"
SAVE_MODEL = "semantic_core/semantic_autoencoder.pth"

features = np.load(FEATURE_PATH)

features = torch.tensor(features, dtype=torch.float32)

dataset = TensorDataset(features, features)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SemanticAutoencoder(input_dim=576, latent_dim=128)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0

    for x, target in loader:
        output = model(x)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), SAVE_MODEL)

print("Autoencoder saved at:", SAVE_MODEL)