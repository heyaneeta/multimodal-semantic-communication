from ollama_client import ask_llama
response = ask_llama(
    "Explain semantic communication in one paragraph."
)

print(response)