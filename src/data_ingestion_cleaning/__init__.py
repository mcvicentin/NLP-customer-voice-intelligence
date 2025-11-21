"""
Data ingestion and text cleaning utilities.
"""

from .fetch_amazon_reviews import fetch_amazon_reviews
from .text_cleaning import clean_text, remove_stopwords

__all__ = [
    "fetch_amazon_reviews",
    "clean_text",
]

