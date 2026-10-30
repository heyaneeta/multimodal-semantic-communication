try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False


class LLMAgent:
    """
    Generates a technical report using a local Ollama model.

    If the Ollama Python package or local Ollama service is unavailable,
    the class returns a fallback report instead of crashing the project.
    """

    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model

    def generate_report(self, data: dict) -> str:
        prediction = data.get("prediction", "Unknown")
        confidence = float(data.get("confidence", 0.0))
        mse = float(data.get("mse", 0.0))
        psnr = float(data.get("psnr", 0.0))
        compression_ratio = float(data.get("compression_ratio", 0.0))
        semantic_quality = float(data.get("semantic_quality", 0.0))
        snr = float(data.get("snr", 0.0))
        channel_status = data.get("channel_status", "Unknown")
        decision = data.get("decision", "No decision")

        prompt = f"""
You are an expert AI communications engineer writing a technical report.

Analyze the following Image Semantic Communication execution data.

TRANSMISSION PROTOCOL:
Semantic 6G Link

IMAGE CLASSIFICATION RESULT:
Prediction: {prediction}
Confidence: {confidence:.2%}

SEMANTIC CORE PERFORMANCE:
MSE: {mse:.6f}
PSNR: {psnr:.2f} dB
Compression Ratio: {compression_ratio:.2f}:1
Semantic Quality Score: {semantic_quality:.2f}%
SNR: {snr:.2f} dB
Channel Status: {channel_status}

AGENT DECISION:
{decision}

Generate a concise professional report with these sections:

1. Classification Summary
2. Transmission Quality
3. Noise Analysis
4. Semantic Reconstruction Quality
5. Decision Explanation
6. Recommendations

Use Markdown formatting.
Do not include introductory conversational text.
"""

        if not OLLAMA_AVAILABLE:
            return self._fallback_report(
                prediction=prediction,
                confidence=confidence,
                mse=mse,
                psnr=psnr,
                compression_ratio=compression_ratio,
                semantic_quality=semantic_quality,
                snr=snr,
                channel_status=channel_status,
                decision=decision,
                reason="The Ollama Python package is not installed."
            )

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt
            )

            generated_text = response.get("response", "").strip()

            if generated_text:
                return generated_text

            return self._fallback_report(
                prediction=prediction,
                confidence=confidence,
                mse=mse,
                psnr=psnr,
                compression_ratio=compression_ratio,
                semantic_quality=semantic_quality,
                snr=snr,
                channel_status=channel_status,
                decision=decision,
                reason="Ollama returned an empty response."
            )

        except Exception as error:
            return self._fallback_report(
                prediction=prediction,
                confidence=confidence,
                mse=mse,
                psnr=psnr,
                compression_ratio=compression_ratio,
                semantic_quality=semantic_quality,
                snr=snr,
                channel_status=channel_status,
                decision=decision,
                reason=str(error)
            )

    def _fallback_report(
        self,
        prediction: str,
        confidence: float,
        mse: float,
        psnr: float,
        compression_ratio: float,
        semantic_quality: float,
        snr: float,
        channel_status: str,
        decision: str,
        reason: str
    ) -> str:
        return f"""
## Classification Summary

The image was classified as **{prediction}** with a confidence of
**{confidence:.2%}**.

## Transmission Quality

The reconstructed semantic embedding produced an MSE of **{mse:.6f}**
and a PSNR of **{psnr:.2f} dB**.

## Noise Analysis

The simulated communication channel operated at an SNR of
**{snr:.2f} dB**. The channel status was reported as
**{channel_status}**.

## Semantic Reconstruction Quality

The semantic quality score was **{semantic_quality:.2f}%**.
The compression ratio was **{compression_ratio:.2f}:1**.

## Decision Explanation

The Agentic AI decision was:

**{decision}**

## Recommendations

Review the channel noise level and packet-loss rate if semantic quality
or confidence becomes low. Retransmission or increased redundancy may
be used under poor channel conditions.

---

*Local LLM report generation was unavailable.*

*Reason: {reason}*
"""
