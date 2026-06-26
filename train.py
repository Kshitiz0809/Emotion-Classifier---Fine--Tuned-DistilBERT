"""
Fine-tune DistilBERT on the dair-ai/emotion dataset (6 classes:
sadness, joy, love, anger, fear, surprise) using the Hugging Face Trainer.

Usage:
    python train.py --push_to_hub --hub_model_id <your-username>/distilbert-emotion
"""

import argparse

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "dair-ai/emotion"

f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"f1": f1["f1"], "accuracy": acc["accuracy"]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="distilbert-emotion")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None, help="e.g. your-username/distilbert-emotion")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(DATASET_NAME)
    label_names = dataset["train"].features["label"].names

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized = dataset.map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized["test"])
    print("Test set metrics:", test_metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Add fine-tuned DistilBERT emotion classifier")


if __name__ == "__main__":
    main()
