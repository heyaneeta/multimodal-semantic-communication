from pathlib import Path
from collections import Counter

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "video_models" / "video_action_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FRAME_INTERVAL = 20

video_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


_model = None
_classes = None


def load_video_model():
    global _model, _classes

    if _model is not None and _classes is not None:
        return _model, _classes

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    _classes = checkpoint["classes"]
    num_classes = len(_classes)

    model = models.mobilenet_v3_small(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    _model = model

    return _model, _classes


def extract_frame_embedding(model, image_tensor):
    """
    Extracts the 576-dimensional MobileNetV3 semantic feature.
    """
    with torch.no_grad():
        features = model.features(image_tensor)
        features = model.avgpool(features)
        embedding = torch.flatten(features, 1)

    return embedding


def analyze_video(video_path):
    model, classes = load_video_model()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_embeddings = []
    frame_predictions = []
    frame_confidences = []

    frame_count = 0
    processed_frames = 0

    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            break

        if frame_count % FRAME_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(rgb_frame)

            image_tensor = (
                video_transform(image)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                logits = model(image_tensor)

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                confidence, predicted_index = torch.max(
                    probabilities,
                    dim=1,
                )

                embedding = extract_frame_embedding(
                    model,
                    image_tensor,
                )

            frame_predictions.append(predicted_index.item())
            frame_confidences.append(confidence.item())
            frame_embeddings.append(embedding.squeeze(0))

            processed_frames += 1

        frame_count += 1

    capture.release()

    if not frame_embeddings:
        raise ValueError("No video frames were processed.")

    stacked_embeddings = torch.stack(frame_embeddings)

    video_embedding = torch.mean(
        stacked_embeddings,
        dim=0,
    )

    majority_prediction = Counter(
        frame_predictions
    ).most_common(1)[0][0]

    matching_confidences = [
        confidence
        for prediction, confidence in zip(
            frame_predictions,
            frame_confidences,
        )
        if prediction == majority_prediction
    ]

    average_confidence = (
        sum(matching_confidences)
        / len(matching_confidences)
    )

    return {
        "prediction": classes[majority_prediction],
        "confidence": average_confidence,
        "embedding": video_embedding.cpu().numpy(),
        "processed_frames": processed_frames,
        "total_frames": frame_count,
    }