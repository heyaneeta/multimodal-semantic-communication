import sys
from pathlib import Path

import torch

from image_models.quality_agent import QualityAgent
from image_models.decision_agent import DecisionAgent
from image_models.coordinator_agent import CoordinatorAgent
from image_models.models import FeatureExtractor, Classifier
from image_models.autoencoder import AutoEncoder


BASE_DIR = Path(__file__).resolve().parent


def verify_agents():
    print("Testing QualityAgent...")

    quality_agent = QualityAgent()

    quality_result = quality_agent.evaluate_quality(
        mse=0.002,
        psnr=26.9,
        semantic_quality=94.5
    )

    print("QualityAgent Output:", quality_result)

    print("\nTesting DecisionAgent...")

    decision_agent = DecisionAgent()

    result_1 = decision_agent.make_decision(
        prediction="Cat",
        confidence=0.92,
        mse=0.002,
        psnr=26.9,
        semantic_quality=94.5,
        snr=15.0
    )

    print("High-quality decision:", result_1)

    result_2 = decision_agent.make_decision(
        prediction="Cat",
        confidence=0.92,
        mse=0.1,
        psnr=10.0,
        semantic_quality=50.0,
        snr=2.0
    )

    print("Noisy-channel decision:", result_2)

    result_3 = decision_agent.make_decision(
        prediction="Cat",
        confidence=0.60,
        mse=0.002,
        psnr=26.9,
        semantic_quality=94.5,
        snr=15.0
    )

    print("Low-confidence decision:", result_3)

    print("\nTesting CoordinatorAgent...")

    coordinator = CoordinatorAgent()

    coordinator_result = coordinator.process_transmission(
        prediction="Cat",
        confidence=0.88,
        mse=0.005,
        psnr=23.0,
        compression_ratio=4.5,
        semantic_quality=85.0,
        snr=12.0,
        channel_status="Clean Channel",
        trigger_llm=False
    )

    print("Coordinator result keys:", coordinator_result.keys())
    print("Quality:", coordinator_result["quality"])
    print("Decision:", coordinator_result["decision"])


def verify_model_architecture():
    print("\nTesting model architecture...")

    feature_extractor = FeatureExtractor()
    autoencoder = AutoEncoder()
    classifier = Classifier()

    dummy_image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        features = feature_extractor(dummy_image)
        compressed, reconstructed = autoencoder(features)
        output = classifier(reconstructed)

    print("Feature shape:", tuple(features.shape))
    print("Compressed shape:", tuple(compressed.shape))
    print("Recovered shape:", tuple(reconstructed.shape))
    print("Classifier output shape:", tuple(output.shape))

    assert features.shape == (1, 576)
    assert compressed.shape == (1, 128)
    assert reconstructed.shape == (1, 576)
    assert output.shape == (1, 2)


def verify_saved_models():
    print("\nChecking trained model files...")

    model_files = {
        "Feature extractor": BASE_DIR / "feature_extractor.pth",
        "Autoencoder": BASE_DIR / "autoencoder.pth",
        "Classifier": BASE_DIR / "classifier.pth"
    }

    missing_files = []

    for name, path in model_files.items():
        if path.exists() and path.stat().st_size > 0:
            print(f"{name}: found")
        else:
            print(f"{name}: missing")
            missing_files.append(str(path))

    if missing_files:
        print("\nTraining is required before checkpoint loading.")
        return False

    feature_extractor = FeatureExtractor()
    autoencoder = AutoEncoder()
    classifier = Classifier()

    feature_extractor.load_state_dict(
        torch.load(
            model_files["Feature extractor"],
            map_location="cpu"
        )
    )

    autoencoder.load_state_dict(
        torch.load(
            model_files["Autoencoder"],
            map_location="cpu"
        )
    )

    classifier.load_state_dict(
        torch.load(
            model_files["Classifier"],
            map_location="cpu"
        )
    )

    print("All trained checkpoints loaded successfully.")
    return True


if __name__ == "__main__":
    try:
        verify_agents()
        verify_model_architecture()
        verify_saved_models()

        print("\nSYSTEM VERIFICATION COMPLETED")

    except Exception as error:
        print("\nVERIFICATION FAILED:", error)
        sys.exit(1)
