# src/rag_system/__init__.py
"""RAG System - A modern Retrieval-Augmented Generation implementation."""

__version__ = "0.1.0"
__author__ = "Your Name"

# optional: you can import specific modules or services here
from .core.config import settings
from .services.embeddings import embedding_service
from .services.ingestion import ingestion_service

__all__ = ["settings", "embedding_service", "ingestion_service", "__version__"]