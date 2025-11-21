# src/models/bert_trainer.py

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
)


class BertReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class BertSentimentTrainer:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    def prepare_dataset(self, df, text_col, label_col, n_samples=40000):
        df_sample = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

        df_sample["label"] = df_sample[label_col].astype(int)
        df_sample["label_id"] = df_sample["label"].map({1: 0, 2: 1})

        train_df, val_df = train_test_split(
            df_sample,
            test_size=0.2,
            random_state=42,
            stratify=df_sample["label_id"],
        )

        train_dataset = BertReviewsDataset(
            train_df[text_col].tolist(),
            train_df["label_id"].tolist(),
            self.tokenizer,
        )

        val_dataset = BertReviewsDataset(
            val_df[text_col].tolist(),
            val_df["label_id"].tolist(),
            self.tokenizer,
        )

        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset, output_dir="./bert_output"):
        args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            learning_rate=3e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=100,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        return trainer

