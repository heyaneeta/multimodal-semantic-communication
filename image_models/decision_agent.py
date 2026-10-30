class DecisionAgent:
    """
    Evaluates semantic transmission metrics and makes a decision based on four rules:
    - Very noisy channel: Recommend Increasing Redundancy
    - Low confidence: Recommend Retransmission
    - High confidence + high PSNR: Accept Transmission
    - Medium confidence: Accept with Warning

    Returns a dictionary with keys: decision, reason, recommended_action.
    """
    def __init__(self, high_confidence: float = 0.85, high_psnr: float = 18.0, 
                 low_confidence: float = 0.70, noisy_snr: float = 3.0):
        self.high_confidence = high_confidence
        self.high_psnr = high_psnr
        self.low_confidence = low_confidence
        self.noisy_snr = noisy_snr

    def make_decision(self, prediction: str, confidence: float, mse: float, 
                      psnr: float, semantic_quality: float, snr: float) -> dict:
        # Rule 1: Very noisy channel
        if snr < self.noisy_snr:
            decision = "Recommend Increasing Redundancy"
            reason = f"Channel Signal-to-Noise Ratio (SNR) is critically low ({snr:.1f} dB), which causes high noise levels."
            recommended_action = "Increase channel coding redundancy, boost transmission power, or use fallback robust modulation."
        
        # Rule 2: Low confidence
        elif confidence < self.low_confidence:
            decision = "Recommend Retransmission"
            reason = f"The classifier's confidence score ({confidence:.2%}) is below the threshold of {self.low_confidence:.0%}, indicating unreliable class recognition."
            recommended_action = "Request automatic repeat transmission (HARQ) of semantic features."
        
        # Rule 3: High confidence + high PSNR
        elif confidence >= self.high_confidence and psnr >= self.high_psnr:
            decision = "Accept Transmission"
            reason = f"Excellent channel conditions (SNR: {snr:.1f} dB) and high prediction accuracy ({confidence:.2%}) with clean feature reconstruction (PSNR: {psnr:.2f} dB)."
            recommended_action = "Verify transmission packet as successfully received and log into transaction storage."
        
        # Rule 4: Medium confidence (and other intermediate states)
        else:
            decision = "Accept with Warning"
            reason = (
                f"Transmission completed but classification confidence is in the medium range ({confidence:.2%}) "
                f"or reconstruction fidelity is moderate (PSNR: {psnr:.2f} dB)."
            )
            recommended_action = "Accept the data block but flag for secondary sanity checking due to potential minor distortion."

        return {
            "decision": decision,
            "reason": reason,
            "recommended_action": recommended_action
        }
