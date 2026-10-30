import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

FRAME_DIR = "datasets/ucf11_frames"
MODEL_PATH = "video_models/video_action_model.pth"

SAVE_FEATURES = "semantic_core/features.npy"
SAVE_LABELS = "semantic_core/labels.npy"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(FRAME_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("Total frames found:", len(dataset))
print("Classes:", dataset.classes)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

classes = checkpoint["classes"]
num_classes = len(classes)

model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model.load_state_dict(checkpoint["model_state"])
model.eval()

feature_extractor = model.features

all_features = []
all_labels = []

print("Extracting semantic features...")

with torch.no_grad():
    for images, labels in tqdm(loader):
        features = feature_extractor(images)
        features = torch.mean(features, dim=[2, 3])

        all_features.append(features.numpy())
        all_labels.append(labels.numpy())

all_features = np.vstack(all_features)
all_labels = np.concatenate(all_labels)

np.save(SAVE_FEATURES, all_features)
np.save(SAVE_LABELS, all_labels)

print("Feature dataset created successfully.")
print("Features shape:", all_features.shape)
print("Labels shape:", all_labels.shape)