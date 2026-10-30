import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        mobilenet = mobilenet_v3_small(weights="DEFAULT")

        self.features = mobilenet.features

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(576,576)

    def forward(self,x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x,1)

        x = self.fc(x)

        return x


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(576,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):

        return self.net(x)