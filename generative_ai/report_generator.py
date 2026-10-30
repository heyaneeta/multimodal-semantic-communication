from agentic_ai.ollama_client import ask_llama


def generate_semantic_report(
    original_meaning,
    recovered_meaning,
    mse_value,
    psnr_value,
    agent_report,
    original_dim=576,
    compressed_dim=128
):
    bandwidth_saving = (1 - compressed_dim / original_dim) * 100

    prompt = f"""
You are a Generative AI technical report writer.

Generate a clear project report for a 6G Semantic Communication system.

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

Agentic AI Analysis:
{agent_report}

Write the report in this format:

Title:
Overview:
Semantic Communication Process:
Performance Analysis:
Agentic AI Decision:
Conclusion:
"""

    return ask_llama(prompt)