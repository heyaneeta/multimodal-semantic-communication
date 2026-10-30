from communication.awgn import add_awgn
from communication.packet_loss import apply_packet_loss


def apply_channel(features, noise_type="awgn", noise_level=0.1):
    if noise_type == "awgn":
        return add_awgn(features, noise_level=noise_level)

    elif noise_type == "packet_loss":
        return apply_packet_loss(features, loss_rate=noise_level)

    else:
        return features