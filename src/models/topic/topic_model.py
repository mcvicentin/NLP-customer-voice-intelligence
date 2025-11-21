"""
Topic Modeling Module
---------------------

Contém classes para:
- LDA (sklearn)
- BERTopic (MiniLM embeddings)
- Visualização (Plotly, Wordcloud)
"""

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
import hdbscan


# ============================================================
# 1. Data Preparation
# ============================================================

def prepare_topic_dataframe(df, text_column="clean_text", n_samples=None, random_state=42):
    """
    Amostra e prepara o dataframe para análise de tópicos.
    """
    if n_samples is None:
        return df.copy().reset_index(drop=True)
    return df.sample(n_samples, random_state=random_state).reset_index(drop=True)


# ============================================================
# 2. LDA Topic Model (Sklearn)
# ============================================================

@dataclass
class LDATopicModel:
    n_topics: int = 10
    max_features: int = 5000

    def fit(self, texts):
        """
        Ajusta TF-IDF + LDA ao corpus.
        """
        self.vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=10,
            stop_words="english",
            max_features=self.max_features
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            learning_method="batch",
            random_state=42
        )

        self.topic_distributions = self.lda.fit_transform(tfidf_matrix)
        return self

    def get_top_words(self, n_top_words=10):
        """
        Retorna principais palavras por tópico.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}

        for idx, comp in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in comp.argsort()[:-n_top_words - 1:-1]]
            topics[idx] = top_words

        return topics


# ============================================================
# 3. BERTopic Wrapper
# ============================================================

@dataclass
class BERTopicModel:
    embedding_model: str = "all-MiniLM-L6-v2"
    nr_topics: str = "auto"
    verbose: bool = True

    def fit(self, texts):
        """
        Ajusta o modelo BERTopic ao corpus.
        """
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=self.nr_topics,
            verbose=self.verbose
        )
        self.topics, self.probs = self.model.fit_transform(texts)
        return self

    def get_topic_info(self):
        return self.model.get_topic_info()

    def get_document_info(self, docs):
        return self.model.get_document_info(docs)

    def visualize_topics(self):
        return self.model.visualize_topics()

    def get_topic(self, topic_id):
        return self.model.get_topic(topic_id)


# ============================================================
# 4. Topic Visualizations
# ============================================================

def plot_topic_frequencies(topic_info, n=15):
    """
    Plotly bar chart das frequências dos principais tópicos.
    """
    df_plot = topic_info[topic_info.Topic >= 0].sort_values("Count", ascending=False).head(n)

    fig = px.bar(
        df_plot,
        x="Count",
        y="Name",
        orientation="h",
        height=600,
        title=f"Top {n} Tópicos mais frequentes"
    )
    return fig


def plot_wordcloud(topic_model, topic_id):
    """
    Wordcloud para um tópico específico.
    """
    words = dict(topic_model.get_topic(topic_id))

    wc = WordCloud(width=900, height=500, background_color="white").generate_from_frequencies(words)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Tópico {topic_id}", fontsize=18)
    plt.show()
