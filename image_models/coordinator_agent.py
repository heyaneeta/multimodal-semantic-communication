from image_models.quality_agent import QualityAgent
from image_models.decision_agent import DecisionAgent
from image_models.llm_agent import LLMAgent

class CoordinatorAgent:
    """
    Coordinates the semantic communication evaluation pipeline.
    Invokes QualityAgent, DecisionAgent, and LLMAgent to analyze and report performance.
    """
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self.quality_agent = QualityAgent()
        self.decision_agent = DecisionAgent()
        self.llm_agent = LLMAgent(model=model_name)

    def process_transmission(self, prediction: str, confidence: float, mse: float, psnr: float, 
                             compression_ratio: float, semantic_quality: float, snr: float, 
                             channel_status: str = "Unknown", trigger_llm: bool = False) -> dict:
        """
        Processes metrics, evaluates quality, decides reception path, and generates reports.
        """
        # Call QualityAgent
        quality_eval = self.quality_agent.evaluate_quality(mse, psnr, semantic_quality)
        
        # Call DecisionAgent
        decision_eval = self.decision_agent.make_decision(
            prediction=prediction,
            confidence=confidence,
            mse=mse,
            psnr=psnr,
            semantic_quality=semantic_quality,
            snr=snr
        )
        
        # Call LLMAgent (Generative Report) on demand
        report = ""
        if trigger_llm:
            data = {
                "prediction": prediction,
                "confidence": confidence,
                "mse": mse,
                "psnr": psnr,
                "compression_ratio": compression_ratio,
                "semantic_quality": semantic_quality,
                "snr": snr,
                "channel_status": channel_status,
                "quality": quality_eval["rating"],
                "decision": decision_eval["decision"]
            }
            report = self.llm_agent.generate_report(data)
        else:
            report = "Ready for report generation. Click the 'Generate Technical Report' button below."

        return {
            "quality": quality_eval,
            "decision": decision_eval,
            "report": report
        }
