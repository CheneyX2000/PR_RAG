# src/rag_system/utils/exceptions.py
"""Custom exceptions for the RAG system"""


class RAGException(Exception):
    """Base exception for RAG system"""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails"""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval fails"""
    pass


class GenerationError(RAGException):
    """Raised when LLM generation fails"""
    pass


class IngestionError(RAGException):
    """Raised when document ingestion fails"""
    pass


class DatabaseError(RAGException):
    """Raised when database operations fail"""
    pass


class ValidationError(RAGException):
    """Raised when input validation fails"""
    pass


class ModelNotFoundError(RAGException):
    """Raised when requested model is not available"""
    pass


class AuthenticationError(RAGException):
    """Raised when authentication fails"""
    pass


class RateLimitError(RAGException):
    """Raised when rate limit is exceeded"""
    pass