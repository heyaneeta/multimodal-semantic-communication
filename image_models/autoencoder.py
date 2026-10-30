import torch
import torch.nn as nn

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(576,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,576)
        )

    def forward(self,x):

        compressed = self.encoder(x)

        reconstructed = self.decoder(compressed)

        return compressed,reconstructed