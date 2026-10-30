import torch

from text_models.src.predict import encode_and_classify_text
from semantic_core.pipeline import process_embedding


def run_text_pipeline(
    text,
    noise_type="awgn",
    noise_level=0.1,
):
    original_result = encode_and_classify_text(text)

    embedding = original_result["embedding"]

    result = process_embedding(
        embedding=embedding,
        noise_type=noise_type,
        noise_level=noise_level,
        modality="Text",
    )

    result["original_prediction"] = original_result["prediction"]
    result["original_confidence"] = original_result["confidence"]

    result.pop("recovered_embedding", None)

    return result