import numpy as np
import torch

from image_models.autoencoder import AutoEncoder
from semantic_core.metrics import (
    calculate_mse,
    calculate_psnr,
    cosine_similarity,
)
from agentic_ai.semantic_agent import semantic_communication_agent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_awgn_torch(vector, noise_level):
    noise = torch.randn_like(vector) * noise_level
    return vector + noise


def apply_packet_loss_torch(vector, loss_rate):
    mask = (torch.rand_like(vector) > loss_rate).float()
    return vector * mask


def process_embedding(
    embedding,
    noise_type="awgn",
    noise_level=0.1,
    modality="Unknown",
):
    original = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    if original.shape[0] != 576:
        raise ValueError(
            f"Expected a 576-dimensional embedding, "
            f"but received {original.shape[0]} dimensions."
        )

    model = AutoEncoder().to(DEVICE)
    model_path = "image_models/autoencoder.pth"

    try:
        state_dict = torch.load(
            model_path,
            map_location=DEVICE,
        )
        model.load_state_dict(state_dict)
        model_loaded = True

    except Exception as error:
        model_loaded = False
        raise RuntimeError(
            f"Could not load autoencoder model: {error}"
        ) from error

    model.eval()

    original_tensor = torch.tensor(
        original,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        compressed = model.encoder(original_tensor)

        if noise_type == "awgn":
            corrupted = add_awgn_torch(
                compressed,
                noise_level,
            )

        elif noise_type == "packet_loss":
            corrupted = apply_packet_loss_torch(
                compressed,
                noise_level,
            )

        else:
            raise ValueError(
                f"Unsupported noise type: {noise_type}"
            )

        recovered = model.decoder(corrupted)

    compressed_np = compressed.squeeze(0).cpu().numpy()
    corrupted_np = corrupted.squeeze(0).cpu().numpy()
    recovered_np = recovered.squeeze(0).cpu().numpy()

    mse = calculate_mse(
        original,
        recovered_np,
    )

    psnr = calculate_psnr(
        original,
        recovered_np,
    )

    cosine = cosine_similarity(
        original,
        recovered_np,
    )

    bandwidth_saving = (
        1 - compressed_np.shape[0] / original.shape[0]
    ) * 100

    report = semantic_communication_agent(
        original_meaning=(
            f"{modality} original 576-dimensional "
            f"semantic embedding"
        ),
        recovered_meaning=(
            f"{modality} recovered 576-dimensional "
            f"semantic embedding"
        ),
        mse_value=float(mse),
        psnr_value=float(psnr),
        original_dim=original.shape[0],
        compressed_dim=compressed_np.shape[0],
    )

    return {
        "modality": modality,
        "original_shape": original.shape,
        "compressed_shape": compressed_np.shape,
        "corrupted_shape": corrupted_np.shape,
        "recovered_shape": recovered_np.shape,
        "mse": float(mse),
        "psnr": float(psnr),
        "cosine_similarity": float(cosine),
        "bandwidth_saving": f"{bandwidth_saving:.2f}%",
        "autoencoder_model_loaded": model_loaded,
        "agentic_ai_report": report,
        "recovered_embedding": recovered_np,
    }