from agent import SemanticAgent
from llm import generate_report

text = "India defeated Australia in the cricket world cup."
prediction = "Sports"
confidence = 0.93
noise_level = 0.10
packet_loss = 0.05

agent = SemanticAgent()

agent_result = agent.run(
    prediction=prediction,
    confidence=confidence,
    noise_level=noise_level,
    packet_loss=packet_loss
)

print("========== Agentic AI ==========")
print(agent_result)

print("\n========== Generative AI ==========")

report = generate_report(
    text=text,
    category=prediction,
    confidence=confidence,
    decision=agent_result["decision"]
)

print(report)