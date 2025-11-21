#!/usr/bin/env python
# coding: utf-8

# ## Topic Modeling w/ LDA & BERTopic
# 
# - Load data
# - Prepare text for topics
# - Create text representations (TF-IDF / embeddings)
# 
# -> for modeling:
# - classic LDA (sklearn)
# - BERTopic 
# - Redução dimensional (UMAP)
# - Visualization & interpretation
# ________________________________
# 
# ------------------
# This notebook requires a **dedicated environment**, since BERTopic dependencies
# (UMAP, hdbscan, numba) with the main environment of the project.
# -------------------

# In[1]:


# ===============================================================
# NOTEBOOK 04 — TOPIC MODELING (LDA + BERTopic)
# --->>>>>>> Ambiente dedicado: "topic"  <<<<<<<<<---------
# ===============================================================

# ===============================================================
# NOTEBOOK 04 — TOPIC MODELING (LDA + BERTopic)
# Detecção automática + tentativa de ativação do ambiente
# ===============================================================

import os
import sys
import subprocess

EXPECTED_ENV = "topic"

print("🔧 Checking environment...\n")

current_env = os.environ.get("CONDA_DEFAULT_ENV", None)
print(f"Current env detected: {current_env}")

# -------------------------------------------------------------------
# 1) Se já estamos no ambiente correto → OK
# -------------------------------------------------------------------
if current_env == EXPECTED_ENV:
    print("✅ Environment OK — running inside 'topic'\n")

else:
    print("⚠️ Not running inside the 'topic' environment.")

    print("➡️ Attempting automatic activation...")
    try:
        # Tentativa de ativar o ambiente (funciona fora do Jupyter)
        subprocess.run(["conda", "activate", EXPECTED_ENV], check=True)
        print("✅ Environment activated! Please restart the kernel.")
    except Exception as e:
        print("\n⛔ Automatic activation failed (expected inside Jupyter).")
        print("   To run this notebook correctly, STOP and run:")
        print("""
        conda activate topic
        jupyter notebook
        """)
    # Para a execução do notebook
    raise SystemExit("Stopping execution: wrong environment.")


# In[3]:


import pandas as pd
import numpy as np
import umap
import hdbscan
from bertopic import BERTopic

print("OK — environment healthy!")

from pathlib import Path

# Caminho dos dados treinados previamente (Amazon Reviews full dataset)
DATA_DIR = Path("../data/processed")

train_path = DATA_DIR / "train_preprocessed.csv"

print("Carregando dados...")
df = pd.read_csv(train_path)

print(df.shape)
df.head()


# In[4]:


# Selecionar subset para topic modeling
N = len(df)#10000   

df_topics = df.sample(N, random_state=42).reset_index(drop=True)
df_topics.head()


# In[5]:


# TF-IDF (for LDA and exploratory anlysis)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_df=0.95,
    min_df=10,
    max_features=5000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(df_topics["clean_text"])

tfidf_matrix.shape


# In[6]:


# LDA
from sklearn.decomposition import LatentDirichletAllocation

NUM_TOPICS = 10

lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS,
    random_state=42,
    learning_method="batch"
)

lda_topics = lda.fit_transform(tfidf_matrix)

print("Shape:", lda_topics.shape)


# In[7]:


# show top words by topic
def display_topics(model, feature_names, n_top_words=12):
    for idx, topic in enumerate(model.components_):
        print(f"\nTopic {idx}:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

feature_names = tfidf.get_feature_names_out()
display_topics(lda, feature_names)


# In[8]:


# Bertropic
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Cria o modelo
topic_model = BERTopic(
    embedding_model="all-MiniLM-L6-v2",   # modelo leve + rápido
    nr_topics="auto",
    verbose=True
)

topics, probs = topic_model.fit_transform(df_topics["text"])

topic_model.get_topic_info().head()


# In[9]:


# visualize topics
topic_fig = topic_model.visualize_topics()
topic_fig.show()


# In[10]:


# table of topics
topic_model.get_topic_info().head(15)


# In[11]:


# inspect one topic
topic_model.get_topic(10)


# In[15]:


# bar plot
import plotly.express as px

df_plot = topic_info[topic_info.Topic >= 0].sort_values("Count", ascending=False).head(15)

fig = px.bar(
    df_plot,
    x="Count",
    y="Name",
    orientation="h",
    title="Top 15 tópicos mais frequentes",
    height=600
)

fig.show()


# In[20]:


# wordcloud/topic
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(topic_id):
    words = dict(topic_model.get_topic(topic_id))
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(words)
    
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Tópico {topic_id}", fontsize=18)
    plt.show()

plot_wordcloud(4)


# In[23]:


docs = df_topics["text"].tolist()

doc_info = topic_model.get_document_info(docs)
doc_info.head()


# In[24]:


# SUMMARy per topic
from transformers import pipeline

summarizer = pipeline("summarization", model="t5-small")

def summarize_topic(topic_id, n_samples=30):
    # Filtra documentos daquele tópico
    subset = doc_info[doc_info["Topic"] == topic_id]
    
    if subset.empty:
        return f"[Tópico {topic_id}] sem documentos suficientes para sumarizar."
    
    # Garante que não tentamos amostrar mais do que existe
    n = min(n_samples, len(subset))
    texts = subset["Document"].sample(n, random_state=42).tolist()
    
    # Junta textos em um bloco só (limitado para não explodir o modelo)
    joined = " ".join(texts)[:4000]  # T5 tem limite de tokens
    
    summary = summarizer(
        joined,
        max_length=120,
        min_length=40,
        do_sample=False
    )
    return summary[0]["summary_text"]

for t in [0, 1, 2, 3, 4]:
    print(f"\n===== TÓPICO {t} =====")
    print(summarize_topic(t))


# In[ ]:




