from agentic_ai.ollama_client import ask_llama

def semantic_communication_agent(
    original_meaning,
    recovered_meaning,
    mse_value,
    psnr_value,
    original_dim=576,
    compressed_dim=128
):
    bandwidth_saving = (1 - compressed_dim / original_dim) * 100

    prompt = f"""
You are an Agentic AI network analyzer for a 6G Semantic Communication system.

Analyze the semantic communication result.

Original Meaning:
{original_meaning}

Recovered Meaning:
{recovered_meaning}

MSE:
{mse_value}

PSNR:
{psnr_value}

Original Semantic Dimension:
{original_dim}

Compressed Latent Dimension:
{compressed_dim}

Bandwidth Saving:
{bandwidth_saving:.2f}%

Tasks:
1. Check whether semantic meaning is preserved.
2. Evaluate transmission quality.
3. Analyze bandwidth saving.
4. Give improvement recommendation.
5. Generate a short technical report.

Return exactly in this format:

Semantic Status:
Transmission Quality:
Bandwidth Saving:
Recommendation:
Technical Report:
"""

    return ask_llama(prompt)