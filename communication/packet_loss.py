import numpy as np

def apply_packet_loss(features, loss_rate=0.2):
    mask = np.random.rand(*features.shape) > loss_rate

    corrupted_features = features * mask

    return corrupted_features