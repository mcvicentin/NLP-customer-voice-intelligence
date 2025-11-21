# src/models/traditional_models.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)


class TfidfSentimentModels:
    def __init__(self, max_features=50000, min_df=5, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
        )

        self.models = {
            "logreg": LogisticRegression(max_iter=200, n_jobs=-1, C=2.0),
            "svm": LinearSVC(C=0.25),
            "nb": MultinomialNB(alpha=0.3)
        }

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def train_model(self, name, X_train, y_train):
        model = self.models[name]
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_val, y_val):
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary"
        )

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": classification_report(y_val, preds)
        }

