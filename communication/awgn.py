import numpy as np

def add_awgn(features, noise_level=0.1):
    noise = np.random.normal(
        loc=0,
        scale=noise_level,
        size=features.shape
    )

    noisy_features = features + noise

    return noisy_features