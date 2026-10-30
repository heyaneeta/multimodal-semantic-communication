from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from image_models.models import FeatureExtractor, Classifier
from semantic_core.pipeline import process_embedding


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_MODEL_DIR = PROJECT_ROOT / "image_models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["Cat", "Dog"]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def load_image_models():
    feature_extractor = FeatureExtractor().to(DEVICE)
    classifier = Classifier().to(DEVICE)

    feature_extractor.load_state_dict(
        torch.load(
            IMAGE_MODEL_DIR / "feature_extractor.pth",
            map_location=DEVICE
        )
    )

    classifier.load_state_dict(
        torch.load(
            IMAGE_MODEL_DIR / "classifier.pth",
            map_location=DEVICE
        )
    )

    feature_extractor.eval()
    classifier.eval()

    return feature_extractor, classifier


def classify_embedding(classifier, embedding_tensor):
    with torch.no_grad():
        output = classifier(embedding_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    index = predicted_index.item()

    return {
        "prediction": CLASS_NAMES[index],
        "confidence": confidence.item()
    }


def run_image_pipeline(
    image_path,
    noise_type="awgn",
    noise_level=0.1
):
    feature_extractor, classifier = load_image_models()

    image = Image.open(image_path).convert("RGB")

    image_tensor = (
        image_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # Extract original 576-dimensional embedding
    with torch.no_grad():
        original_embedding_tensor = feature_extractor(
            image_tensor
        )

    # Classify original embedding
    original_result = classify_embedding(
        classifier,
        original_embedding_tensor
    )

    original_embedding = (
        original_embedding_tensor
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # Compression, channel, recovery and metrics
    result = process_embedding(
        embedding=original_embedding,
        noise_type=noise_type,
        noise_level=noise_level,
        modality="Image"
    )

    # Convert recovered NumPy embedding back to tensor
    recovered_embedding_tensor = torch.tensor(
        result["recovered_embedding"],
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    # Classify recovered embedding
    recovered_result = classify_embedding(
        classifier,
        recovered_embedding_tensor
    )

    semantic_preserved = (
        original_result["prediction"]
        == recovered_result["prediction"]
    )

    result["original_prediction"] = original_result["prediction"]
    result["original_confidence"] = original_result["confidence"]

    result["recovered_prediction"] = recovered_result["prediction"]
    result["recovered_confidence"] = recovered_result["confidence"]

    result["semantic_preserved"] = semantic_preserved

    # Remove the large vector before displaying in Streamlit
    result.pop("recovered_embedding", None)

    return result