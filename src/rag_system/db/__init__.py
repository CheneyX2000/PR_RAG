# src/rag_system/db/__init__.py
"""Database module with models and operations."""

from .models import Document, DocumentChunk, ChunkEmbedding
from .pgvector import db, init_database

__all__ = [
    "Document",
    "DocumentChunk", 
    "ChunkEmbedding",
    "db",
    "init_database",
]