from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "text_models" / "models" / "bert_semantic_model"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology",
}


class TextSemanticEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert_model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            output_hidden_states=True,
        )

        self.projection = nn.Linear(768, 576)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        cls_embedding = outputs.hidden_states[-1][:, 0, :]
        semantic_embedding = self.projection(cls_embedding)

        return outputs.logits, semantic_embedding


_tokenizer = None
_encoder = None


def load_text_model():
    global _tokenizer, _encoder

    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    if _encoder is None:
        _encoder = TextSemanticEncoder().to(DEVICE)
        _encoder.eval()

    return _tokenizer, _encoder


def encode_and_classify_text(text):
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty.")

    tokenizer, encoder = load_text_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits, semantic_embedding = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    prediction_index = predicted_index.item()

    return {
        "prediction": LABELS[prediction_index],
        "confidence": confidence.item(),
        "embedding": (
            semantic_embedding
            .squeeze(0)
            .cpu()
            .numpy()
        ),
    }