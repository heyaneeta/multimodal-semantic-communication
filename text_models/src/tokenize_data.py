from datasets import load_from_disk
from transformers import BertTokenizer

# Load dataset
dataset = load_from_disk("data/ag_news")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("Tokenizing dataset...")

tokenized_dataset = dataset.map(tokenize, batched=True)

print("\nTokenization Complete!")

print(tokenized_dataset)