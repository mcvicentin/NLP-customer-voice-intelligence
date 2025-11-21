# src/utils/__init__.py

from .config import (
    PROJECT_ROOT,
    DATA_DIR, RAW_DIR, PROCESSED_DIR,
    MODELS_DIR, LOG_DIR
)

from .logging_utils import get_logger
from .topic_evaluation import topic_distribution, top_docs

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR", "RAW_DIR", "PROCESSED_DIR",
    "MODELS_DIR", "LOG_DIR",
    "get_logger",
    "topic_distribution",
    "top_docs",
]

