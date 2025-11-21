# __init__.py
"""
Spark NLP pipelines and distributed feature engineering components.
"""

from .spark_session import run_full_embedding_pipeline
from .clean_text_spark import run_spark_cleaning
from .nlp_pipeline_spark import (
    build_cleaning_pipeline,
    build_embedding_pipeline
)
from .feature_engineering_spark import (
    extract_token_embeddings,
    explode_embeddings,
    compute_embedding_means,
    vectorize_mean_embeddings
)

__all__ = [
    "run_full_embedding_pipeline",
    "run_spark_cleaning",
    "build_cleaning_pipeline",
    "build_embedding_pipeline",
    "extract_token_embeddings",
    "explode_embeddings",
    "compute_embedding_means",
    "vectorize_mean_embeddings",
]
