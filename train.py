from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# =========================
# Load Dataset
# =========================

with open(
    "D:/EduExpose/inter/dataset/IITB.en-hi.en",
    "r",
    encoding="utf-8"
) as f:
    english_sentences = f.readlines()

with open(
    "D:/EduExpose/inter/dataset/IITB.en-hi.hi",
    "r",
    encoding="utf-8"
) as f:
    hindi_sentences = f.readlines()

# Clean text
english_sentences = [x.strip() for x in english_sentences]
hindi_sentences = [x.strip() for x in hindi_sentences]

# =========================
# Use only first 5000 pairs
# =========================

english_sentences = english_sentences[:100]
hindi_sentences = hindi_sentences[:100]

print("Training Samples:", len(english_sentences))

# =========================
# Create Dataset
# =========================

data = {
    "english": english_sentences,
    "hindi": hindi_sentences
}

dataset = Dataset.from_dict(data)

# Train/Validation Split
dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# =========================
# Load Model
# =========================

model_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# =========================
# Tokenization
# =========================

def preprocess(examples):

    model_inputs = tokenizer(
        examples["english"],
        max_length=128,
        truncation=True
    )

    labels = tokenizer(
        examples["hindi"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

train_dataset = train_dataset.map(
    preprocess,
    batched=True
)

eval_dataset = eval_dataset.map(
    preprocess,
    batched=True
)

# Remove text columns
train_dataset = train_dataset.remove_columns(
    ["english", "hindi"]
)

eval_dataset = eval_dataset.remove_columns(
    ["english", "hindi"]
)

# =========================
# Data Collator
# =========================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# =========================
# Training Arguments
# =========================

training_args = Seq2SeqTrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_steps=50,
    report_to="none"
)

# =========================
# Trainer
# =========================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)
# =========================
# Train
# =========================

trainer.train()

# =========================
# Save Model
# =========================

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training Completed Successfully!")