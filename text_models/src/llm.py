import ollama


def explain_prediction(text, category, confidence=None, agent_decision=None):
    """
    Uses Ollama (Llama 3) to generate a natural-language explanation
    for the semantic communication prediction.
    """

    prompt = f"""
You are an AI assistant for a 6G Semantic Communication System.

Input Text:
{text}

Predicted Category:
{category}

Confidence Score:
{confidence if confidence is not None else "Not Available"}

Agent Decision:
{agent_decision if agent_decision is not None else "No decision provided"}

Explain:

1. Why this text belongs to the predicted category.
2. Whether the prediction is reliable.
3. Why the agent made this decision.

Keep the explanation simple.
Maximum 5 sentences.
"""

    try:
        response = ollama.chat(
            model="llama3:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"Error while generating explanation: {e}"


def generate_report(text, category, confidence, decision):
    """
    Generates the final report shown to the user.
    """

    explanation = explain_prediction(
        text=text,
        category=category,
        confidence=confidence,
        agent_decision=decision
    )

    report = f"""
=========================================
      6G SEMANTIC COMMUNICATION REPORT
=========================================

Input Text:
{text}

-----------------------------------------

Predicted Category:
{category}

-----------------------------------------

Confidence:
{confidence}

-----------------------------------------

Agentic AI Decision:
{decision}

-----------------------------------------

Generative AI Explanation:

{explanation}

=========================================
"""

    return report


# Standalone testing
if __name__ == "__main__":

    sample_text = "India defeated Australia in the cricket world cup."

    category = "Sports"

    confidence = 0.93

    decision = "Prediction accepted"

    print(
        generate_report(
            sample_text,
            category,
            confidence,
            decision
        )
    )