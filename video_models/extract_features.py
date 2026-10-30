import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

MODEL_PATH = "video_models/video_action_model.pth"

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

# This part extracts semantic features
feature_extractor = model.features

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

video_path = input("Enter video path: ")

cap = cv2.VideoCapture(video_path)

all_features = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % 20 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = feature_extractor(image)

            # Convert feature map to 1D semantic vector
            features = torch.mean(features, dim=[2, 3])

            all_features.append(features.numpy()[0])

    frame_count += 1

cap.release()

if len(all_features) == 0:
    print("No frames processed.")
else:
    video_semantic_vector = np.mean(all_features, axis=0)

    print("Semantic feature vector extracted successfully.")
    print("Feature vector shape:", video_semantic_vector.shape)
    print("First 10 feature values:")
    print(video_semantic_vector[:10])

    np.save("video_models/semantic_features.npy", video_semantic_vector)

    print("Saved at: video_models/semantic_features.npy")