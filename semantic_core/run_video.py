import torch

from semantic_core.pipeline import process_embedding
from video_models.predict_video import (
    analyze_video,
    load_video_model,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def classify_video_embedding(embedding):
    model, classes = load_video_model()

    embedding_tensor = torch.tensor(
        embedding,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model.classifier(embedding_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

    return {
        "prediction": classes[predicted_index.item()],
        "confidence": confidence.item(),
    }


def run_video_pipeline(
    video_path,
    noise_type="awgn",
    noise_level=0.1,
):
    original_result = analyze_video(video_path)

    original_embedding = original_result["embedding"]

    result = process_embedding(
        embedding=original_embedding,
        noise_type=noise_type,
        noise_level=noise_level,
        modality="Video",
    )

    recovered_result = classify_video_embedding(
        result["recovered_embedding"]
    )

    result["original_prediction"] = (
        original_result["prediction"]
    )

    result["original_confidence"] = (
        original_result["confidence"]
    )

    result["recovered_prediction"] = (
        recovered_result["prediction"]
    )

    result["recovered_confidence"] = (
        recovered_result["confidence"]
    )

    result["semantic_preserved"] = (
        original_result["prediction"]
        == recovered_result["prediction"]
    )

    result["processed_frames"] = (
        original_result["processed_frames"]
    )

    result["total_frames"] = (
        original_result["total_frames"]
    )

    result.pop("recovered_embedding", None)

    return result