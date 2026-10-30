from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from image_models.models import FeatureExtractor, Classifier
from image_models.autoencoder import AutoEncoder


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "image"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)
print("Dataset path:", DATASET_DIR)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=str(DATASET_DIR),
    transform=transform
)

print("Images found:", len(dataset))
print("Classes:", dataset.classes)

if len(dataset.classes) != 2:
    raise ValueError(
        f"Expected 2 image classes, but found {len(dataset.classes)}: "
        f"{dataset.classes}"
    )

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

feature_extractor = FeatureExtractor().to(DEVICE)
autoencoder = AutoEncoder().to(DEVICE)
classifier = Classifier(num_classes=len(dataset.classes)).to(DEVICE)

classification_loss = nn.CrossEntropyLoss()
reconstruction_loss = nn.MSELoss()

optimizer = optim.Adam(
    list(feature_extractor.parameters())
    + list(autoencoder.parameters())
    + list(classifier.parameters()),
    lr=0.001
)

epochs = 10

print("Training started")

for epoch in range(epochs):
    feature_extractor.train()
    autoencoder.train()
    classifier.train()

    total_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch + 1}/{epochs}")

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = feature_extractor(images)

        compressed, _ = autoencoder(features)

        noise = torch.randn_like(compressed) * 0.1
        noisy_features = compressed + noise

        recovered = autoencoder.decoder(noisy_features)
        outputs = classifier(recovered)

        cls_loss = classification_loss(outputs, labels)
        rec_loss = reconstruction_loss(recovered, features)
        loss = cls_loss + rec_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        if batch_index % 50 == 0:
            print(
                f"Batch {batch_index}/{len(loader)} "
                f"Loss={loss.item():.4f}"
            )

    accuracy = 100 * correct / max(total, 1)

    print(
        f"Epoch {epoch + 1} complete "
        f"Loss={total_loss:.4f} "
        f"Accuracy={accuracy:.2f}%"
    )

torch.save(
    feature_extractor.state_dict(),
    BASE_DIR / "feature_extractor.pth"
)

torch.save(
    autoencoder.state_dict(),
    BASE_DIR / "autoencoder.pth"
)

torch.save(
    classifier.state_dict(),
    BASE_DIR / "classifier.pth"
)

print("\nTraining complete")
print("Saved:")
print(BASE_DIR / "feature_extractor.pth")
print(BASE_DIR / "autoencoder.pth")
print(BASE_DIR / "classifier.pth")