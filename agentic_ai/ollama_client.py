import ollama


MODEL_NAME = "qwen2.5:3b"


def ask_llama(prompt: str) -> str:
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
        )

        generated_text = response.get("response", "").strip()

        if not generated_text:
            return "Ollama returned an empty response."

        return generated_text

    except Exception as error:
        return f"Ollama Error: {error}"