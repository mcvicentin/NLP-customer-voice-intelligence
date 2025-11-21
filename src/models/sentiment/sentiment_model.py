# src/models/sentiment_model.py

import pandas as pd
from pathlib import Path

from data_ingestion_cleaning.text_cleaning import clean_text, remove_stopwords
from models.traditional_models import TfidfSentimentModels
from models.bert_trainer import BertSentimentTrainer


def load_fasttext(path, max_rows=None):
    import bz2
    texts, labels = [], []

    with bz2.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break

            try:
                label, text = line.split(" ", 1)
            except ValueError:
                continue

            labels.append(label.replace("__label__", "").strip())
            texts.append(text.strip())

    return pd.DataFrame({"label": labels, "text": texts})


def prepare_data(df):
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_text_no_stop"] = df["clean_text"].apply(remove_stopwords)
    return df


def train_traditional(df):
    X = df["clean_text_no_stop"].values
    y = df["label"].astype(int).values

    # split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = TfidfSentimentModels()

    X_train_vec = models.fit_transform(X_train)
    X_val_vec   = models.transform(X_val)

    results = {}
    for name in ["logreg", "svm", "nb"]:
        mdl = models.train_model(name, X_train_vec, y_train)
        results[name] = models.evaluate(mdl, X_val_vec, y_val)

    return results


def train_bert(df):
    bert = BertSentimentTrainer()

    train_ds, val_ds = bert.prepare_dataset(
        df, text_col="clean_text_no_stop", label_col="label"
    )

    trainer = bert.train(train_ds, val_ds)
    return trainer.evaluate()

