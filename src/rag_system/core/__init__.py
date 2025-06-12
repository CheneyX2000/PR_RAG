# src/rag_system/core/__init__.py
"""Core module containing configuration and dependencies."""

from .config import settings
from .dependencies import get_db, get_cache

__all__ = ["settings", "get_db", "get_cache"]