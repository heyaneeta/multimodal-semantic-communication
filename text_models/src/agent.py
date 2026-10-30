class SemanticAgent:
    def observe(self, prediction, confidence, noise_level, packet_loss):
        return {
            "prediction": prediction,
            "confidence": confidence,
            "noise_level": noise_level,
            "packet_loss": packet_loss
        }

    def analyze(self, observation):
        risk_score = 0
        reasons = []

        # Analyze confidence
        if observation["confidence"] < 0.60:
            risk_score += 3
            reasons.append("Low confidence detected")
        elif observation["confidence"] < 0.80:
            risk_score += 1
            reasons.append("Medium confidence detected")

        # Analyze channel noise
        if observation["noise_level"] > 0.40:
            risk_score += 2
            reasons.append("High channel noise detected")
        elif observation["noise_level"] > 0.20:
            risk_score += 1
            reasons.append("Medium channel noise detected")

        # Analyze packet loss
        if observation["packet_loss"] > 0.30:
            risk_score += 2
            reasons.append("High packet loss detected")
        elif observation["packet_loss"] > 0.10:
            risk_score += 1
            reasons.append("Medium packet loss detected")

        if not reasons:
            reasons.append("Transmission quality is good")

        return risk_score, reasons

    def decide(self, observation, risk_score):
        if risk_score >= 5:
            return "Retransmit semantic features"

        elif observation["confidence"] < 0.60:
            return "Apply stronger semantic recovery"

        elif observation["noise_level"] > 0.40:
            return "Activate noise-resistant recovery"

        elif observation["packet_loss"] > 0.30:
            return "Reduce compression and resend"

        elif risk_score >= 2:
            return "Perform semantic verification"

        else:
            return "Accept prediction"

    def act(self, decision):
        actions = {
            "Retransmit semantic features":
                "The agent requests retransmission because the received semantic meaning is unreliable.",

            "Apply stronger semantic recovery":
                "The agent applies stronger semantic recovery because prediction confidence is low.",

            "Activate noise-resistant recovery":
                "The agent activates recovery to reduce the effect of channel noise.",

            "Reduce compression and resend":
                "The agent reduces compression so that more semantic information is preserved.",

            "Perform semantic verification":
                "The agent verifies the prediction before accepting the result.",

            "Accept prediction":
                "The agent accepts the prediction because confidence and channel quality are good."
        }

        return actions.get(decision, "No action selected")

    def run(self, prediction, confidence, noise_level, packet_loss):
        observation = self.observe(
            prediction,
            confidence,
            noise_level,
            packet_loss
        )

        risk_score, reasons = self.analyze(observation)

        decision = self.decide(observation, risk_score)

        action = self.act(decision)

        return {
            "observation": observation,
            "risk_score": risk_score,
            "reasons": reasons,
            "decision": decision,
            "action": action
        }