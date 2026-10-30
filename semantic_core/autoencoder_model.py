import torch
import torch.nn as nn

class SemanticAutoencoder(nn.Module):
    def __init__(self, input_dim=576, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)