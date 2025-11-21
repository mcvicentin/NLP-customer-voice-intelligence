"""
Summarization Module
--------------------

Usa T5-small para gerar sumarizações dos tópicos encontrados por BERTopic.
"""

from transformers import pipeline
import pandas as pd


class TopicSummarizer:
    def __init__(self, model_name="t5-small"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize_topic(self, topic_id, doc_info, n_samples=30):
        """
        Gera um sumário curto para um tópico do BERTopic.
        """
        subset = doc_info[doc_info["Topic"] == topic_id]

        if subset.empty:
            return f"[Tópico {topic_id}] sem documentos suficientes para sumarizar."

        n = min(n_samples, len(subset))
        texts = subset["Document"].sample(n, random_state=42).tolist()

        joined = " ".join(texts)[:4000]  # limita tamanho para T5

        summary = self.summarizer(
            joined,
            max_length=120,
            min_length=40,
            do_sample=False
        )

        return summary[0]["summary_text"]
