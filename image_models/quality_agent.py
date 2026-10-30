class QualityAgent:
    """
    Evaluates the semantic communication quality of the transmission based on:
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR)
    - Semantic Quality (Cosine Similarity percentage between original and recovered features)

    Outputs one of: Excellent, Good, Fair, Poor, along with a technical explanation.
    """
    def __init__(self, high_similarity: float = 90.0, mid_similarity: float = 80.0, low_similarity: float = 70.0):
        self.high_similarity = high_similarity
        self.mid_similarity = mid_similarity
        self.low_similarity = low_similarity

    def evaluate_quality(self, mse: float, psnr: float, semantic_quality: float) -> dict:
        """
        Evaluate transmission metrics and output quality classification and reasoning.
        """
        # Determine rating based on semantic similarity and PSNR
        if semantic_quality >= self.high_similarity and psnr >= 20.0:
            rating = "Excellent"
            explanation = (
                f"Excellent transmission quality. Semantic similarity is high ({semantic_quality:.2f}%) "
                f"and distortion is minimal (PSNR: {psnr:.2f} dB, MSE: {mse:.6f}). The reconstructed semantic "
                f"representation retains near-perfect fidelity."
            )
        elif semantic_quality >= self.mid_similarity and psnr >= 15.0:
            rating = "Good"
            explanation = (
                f"Good transmission quality. The channel induced minor signal distortion, but the reconstructed "
                f"features preserve {semantic_quality:.2f}% similarity (PSNR: {psnr:.2f} dB, MSE: {mse:.6f}). "
                f"Core semantic properties are intact."
            )
        elif semantic_quality >= self.low_similarity and psnr >= 10.0:
            rating = "Fair"
            explanation = (
                f"Fair transmission quality. Noticeable degradation in signal fidelity (PSNR: {psnr:.2f} dB) "
                f"and feature similarity ({semantic_quality:.2f}%). Some semantic parameters may be distorted, "
                f"but classification features are still recognizable."
            )
        else:
            rating = "Poor"
            explanation = (
                f"Poor transmission quality. Severe noise or high distortion detected (PSNR: {psnr:.2f} dB, "
                f"MSE: {mse:.6f}). Semantic similarity is insufficient ({semantic_quality:.2f}%), compromising "
                f"overall signal integrity."
            )

        return {
            "rating": rating,
            "explanation": explanation
        }
