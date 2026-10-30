from datasets import load_from_disk
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Load dataset
dataset = load_from_disk("data/ag_news")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Rename label column
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Keep only required columns
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# Training configuration
training_args = TrainingArguments(
    output_dir="models",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=500,
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

print("Starting BERT Training...")

trainer.train()

print("Training Finished!")

model.save_pretrained("models/bert_semantic_model")
tokenizer.save_pretrained("models/bert_semantic_model")

print("Model Saved Successfully!")