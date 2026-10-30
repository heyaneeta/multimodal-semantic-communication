import numpy as np


def calculate_mse(original, recovered):
    return np.mean((original - recovered) ** 2)


def calculate_psnr(original, recovered):
    mse = calculate_mse(original, recovered)

    if mse == 0:
        return float("inf")

    max_value = np.max(original)

    return 20 * np.log10(max_value / np.sqrt(mse))


def cosine_similarity(original, recovered):
    numerator = np.dot(original, recovered)

    denominator = (
        np.linalg.norm(original)
        * np.linalg.norm(recovered)
    )

    if denominator == 0:
        return 0

    return numerator / denominator