#!/usr/bin/env python
# coding: utf-8

# # In this notebook: 
# 
# - Upload and clean data
# - Split train/validation
# - TF-IDF
# - Baseline models (LogReg, SVM, Naive Bayes)
# - Final comparison
# - Save best model

# In[2]:


import pandas as pd
from pathlib import Path

import re
import contractions
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stopwords_en = set(ENGLISH_STOP_WORDS)

DATA_RAW = Path("../data/raw")
TRAIN_PATH = DATA_RAW / "train.ft.txt.bz2"

print("Train exists:", TRAIN_PATH.exists())


# In[3]:


# clean text (as in notebook 01_eda_reviews.ipynb)
def clean_text(text):
    text = contractions.fix(text)         # don't → do not
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\b\d+\b", " ", text)  # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_en and len(t) > 1]
    return " ".join(tokens)


# In[4]:


import bz2

def load_ft_dataset(path, max_rows=None):
    texts = []
    labels = []
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


# In[5]:


# upload data
df = load_ft_dataset(TRAIN_PATH, max_rows=50000)

df["clean_text"] = df["text"].apply(clean_text)
df["clean_text_no_stop"] = df["clean_text"].apply(remove_stopwords)

# save
df.to_csv('../data/processed/train_preprocessed.csv')

df.head()


# # pre- modeling:
# - Split sample (train/val)
# - Create TF-IDF matrix
# - prepare variables 

# In[5]:


from sklearn.model_selection import train_test_split

# Splt sample 
X = df["clean_text_no_stop"].values
y = df["label"].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

len(X_train), len(X_val)


# In[6]:


# Create TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=50000,     # n_features
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=5,               # remove palavras raras
    sublinear_tf=True       # TF sublinear melhora LR/SVM
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)

X_train_tfidf.shape, X_val_tfidf.shape


# In[7]:


# show some features:
tfidf.get_feature_names_out()[:20]


# # Let's repeat logistic regression as in notebook 01_eda_reviews

# In[8]:


# train model
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(
    max_iter=200,
    n_jobs=-1,
    C=2.0,
)

clf_lr.fit(X_train_tfidf, y_train)


# In[9]:


# Predict & metrics
from sklearn.metrics import accuracy_score, classification_report

y_pred_lr = clf_lr.predict(X_val_tfidf)

acc_lr = accuracy_score(y_val, y_pred_lr)
print(f"Logistic Regression - Validation Accuracy: {acc_lr:.4f}\n")

print("Classification report (Logistic Regression):")
print(classification_report(y_val, y_pred_lr, digits=4))


# In[10]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_lr = confusion_matrix(y_val, y_pred_lr)

plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (TF-IDF)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[11]:


# save result
results = {}
results["logreg"] = {
    "accuracy": acc_lr,
    "cm": cm_lr,
}
results


# # Cool! Nice results. 
# # Now, let's test the SVM...

# In[12]:


from sklearn.svm import LinearSVC

clf_svm = LinearSVC(
    C=0.25,         # regularização (podemos tunar depois)
)

clf_svm.fit(X_train_tfidf, y_train)


# In[13]:


y_pred_svm = clf_svm.predict(X_val_tfidf)

from sklearn.metrics import accuracy_score, classification_report

acc_svm = accuracy_score(y_val, y_pred_svm)
print(f"Linear SVM - Validation Accuracy: {acc_svm:.4f}\n")

print("Classification report (Linear SVM):")
print(classification_report(y_val, y_pred_svm, digits=4))


# In[14]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_svm = confusion_matrix(y_val, y_pred_svm)

plt.figure(figsize=(5,4))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Linear SVM (TF-IDF)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# # Logistic reg vs SVM: Similar results!
# 
# # Let's try one more: Multinomial naive Bayes

# In[15]:


from sklearn.naive_bayes import MultinomialNB

clf_nb = MultinomialNB(alpha=0.3)   # alpha pode ser tunado depois
clf_nb.fit(X_train_tfidf, y_train)

y_pred_nb = clf_nb.predict(X_val_tfidf)


# In[16]:


acc_nb = accuracy_score(y_val, y_pred_nb)
print(f"Naive Bayes - Validation Accuracy: {acc_nb:.4f}\n")

print("Classification report (Naive Bayes):")
print(classification_report(y_val, y_pred_nb, digits=4))


# In[17]:


cm_nb = confusion_matrix(y_val, y_pred_nb)

plt.figure(figsize=(5,4))
sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix - Naive Bayes (TF-IDF)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# # OK! Results are similar between thse models. 
# # Let's try more sophisticated models now...

# In[18]:


# Setup DistilBERT

# Imports & Label mapping
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[40]:


# change labels from {1,2} to {0,1} -> for HuggingFace
df_bert = df[["clean_text_no_stop", "label"]].copy()

# convert labels to integers
df_bert["label"] = df_bert["label"].astype(int)

label_map = {1: 0, 2: 1}   # 0 = negative, 1 = positive
df_bert["label_id"] = df_bert["label"].map(label_map)

df_bert["label_id"].value_counts()


# In[57]:


# Prepara dados p BERT
N_BERT = 40000  # pode aumentar depois se estiver tranquilo

df_bert_small = df_bert.sample(n=N_BERT, random_state=42).reset_index(drop=True)
df_bert_small.head()
len(df_bert_small)


# In[59]:


# Split sample
train_df_bert, val_df_bert = train_test_split(
    df_bert_small,
    test_size=0.2,
    random_state=42,
    stratify=df_bert_small["label_id"],
)

len(train_df_bert), len(val_df_bert)


# In[60]:


# carrega tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# # Next, we transform the DF in a HuggingFace dataset compatible with the Trainer.

# In[61]:


# class: receive tokenizer; tokenize ceach text; return  input_ids, attention_mask and labels
class BertReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text   = self.texts[idx]
        label  = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


# In[62]:


# Prepare training sample
train_dataset = BertReviewsDataset(
    train_df_bert["clean_text_no_stop"].tolist(),
    train_df_bert["label_id"].tolist(),
    tokenizer,
    max_len=128
)

val_dataset = BertReviewsDataset(
    val_df_bert["clean_text_no_stop"].tolist(),
    val_df_bert["label_id"].tolist(),
    tokenizer,
    max_len=128
)


# In[63]:


# upload base model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
).to(device)


# In[64]:


# training arguments
training_args = TrainingArguments(
    output_dir="./bert_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,  # deixa mais leve em CPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,             #  ← aumenta para 3
    learning_rate=3e-5,             #  ← LR mais forte
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
)


# In[65]:


# compute metrics
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    
    preds = np.argmax(logits, axis=1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# In[66]:


# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,   
)


# In[67]:


# treina
trainer.train()


# In[68]:


eval_results = trainer.evaluate()
eval_results


# # Classical models shown good initial performance, and a more modern one--DistilBERT--shown similar results, but with good potential to improve, giving the limited computational resources. We will test it again later in a GPU environment with full dataset.
