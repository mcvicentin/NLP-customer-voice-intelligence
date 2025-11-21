# src/utils/topic_evaluation.py
import pandas as pd

def topic_distribution(df, column="Topic"):
    """
    Retorna distribuição de tópicos ordenada.
    """
    return df[column].value_counts().rename("count")

def top_docs(df, topic_id, text_column="Document", topic_column="Topic", n=5):
    """
    Retorna documentos mais representativos de um tópico.
    """
    subset = df[df[topic_column] == topic_id]
    return subset.sample(min(n, len(subset)))
