# tests/test_exceptions.py
"""
Comprehensive tests for custom exceptions.
"""

import pytest
from src.rag_system.utils.exceptions import (
    RAGException,
    EmbeddingError,
    RetrievalError,
    GenerationError,
    IngestionError,
    DatabaseError,
    ValidationError,
    ModelNotFoundError,
    AuthenticationError,
    RateLimitError
)


class TestExceptions:
    """Test custom exception classes"""
    
    def test_base_exception(self):
        """Test base RAGException"""
        with pytest.raises(RAGException) as exc_info:
            raise RAGException("Base error message")
        
        assert str(exc_info.value) == "Base error message"
        assert isinstance(exc_info.value, Exception)
    
    def test_embedding_error(self):
        """Test EmbeddingError"""
        with pytest.raises(EmbeddingError) as exc_info:
            raise EmbeddingError("Embedding generation failed")
        
        assert str(exc_info.value) == "Embedding generation failed"
        assert isinstance(exc_info.value, RAGException)
    
    def test_retrieval_error(self):
        """Test RetrievalError"""
        with pytest.raises(RetrievalError) as exc_info:
            raise RetrievalError("Search failed")
        
        assert str(exc_info.value) == "Search failed"
        assert isinstance(exc_info.value, RAGException)
    
    def test_generation_error(self):
        """Test GenerationError"""
        with pytest.raises(GenerationError) as exc_info:
            raise GenerationError("LLM generation failed")
        
        assert str(exc_info.value) == "LLM generation failed"
        assert isinstance(exc_info.value, RAGException)
    
    def test_ingestion_error(self):
        """Test IngestionError"""
        with pytest.raises(IngestionError) as exc_info:
            raise IngestionError("Document ingestion failed")
        
        assert str(exc_info.value) == "Document ingestion failed"
        assert isinstance(exc_info.value, RAGException)
    
    def test_database_error(self):
        """Test DatabaseError"""
        with pytest.raises(DatabaseError) as exc_info:
            raise DatabaseError("Database connection failed")
        
        assert str(exc_info.value) == "Database connection failed"
        assert isinstance(exc_info.value, RAGException)
    
    def test_validation_error(self):
        """Test ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid input data")
        
        assert str(exc_info.value) == "Invalid input data"
        assert isinstance(exc_info.value, RAGException)
    
    def test_model_not_found_error(self):
        """Test ModelNotFoundError"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            raise ModelNotFoundError("Model 'xyz' not found")
        
        assert str(exc_info.value) == "Model 'xyz' not found"
        assert isinstance(exc_info.value, RAGException)
    
    def test_authentication_error(self):
        """Test AuthenticationError"""
        with pytest.raises(AuthenticationError) as exc_info:
            raise AuthenticationError("Invalid API key")
        
        assert str(exc_info.value) == "Invalid API key"
        assert isinstance(exc_info.value, RAGException)
    
    def test_rate_limit_error(self):
        """Test RateLimitError"""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError("Rate limit exceeded")
        
        assert str(exc_info.value) == "Rate limit exceeded"
        assert isinstance(exc_info.value, RAGException)
    
    def test_exception_hierarchy(self):
        """Test exception inheritance hierarchy"""
        # All custom exceptions should inherit from RAGException
        exceptions = [
            EmbeddingError("test"),
            RetrievalError("test"),
            GenerationError("test"),
            IngestionError("test"),
            DatabaseError("test"),
            ValidationError("test"),
            ModelNotFoundError("test"),
            AuthenticationError("test"),
            RateLimitError("test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, RAGException)
            assert isinstance(exc, Exception)
    
    def test_exception_with_context(self):
        """Test exceptions with additional context"""
        try:
            # Simulate an operation that fails
            embedding_dim = 1536
            actual_dim = 768
            raise ValidationError(
                f"Embedding dimension mismatch: expected {embedding_dim}, got {actual_dim}"
            )
        except ValidationError as e:
            assert "1536" in str(e)
            assert "768" in str(e)
            assert "mismatch" in str(e)
    
    def test_exception_chaining(self):
        """Test exception chaining"""
        original_error = ValueError("Original error")
        
        try:
            try:
                raise original_error
            except ValueError as e:
                raise DatabaseError("Database operation failed") from e
        except DatabaseError as e:
            assert str(e) == "Database operation failed"
            assert e.__cause__ == original_error
    
    def test_exception_in_context_manager(self):
        """Test exceptions in context managers"""
        class TestContextManager:
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is ValidationError:
                    # Handle validation errors
                    return False  # Propagate the exception
                return False
        
        with pytest.raises(ValidationError):
            with TestContextManager():
                raise ValidationError("Context manager test")
    
    def test_exception_error_codes(self):
        """Test that exceptions can carry error codes"""
        class ErrorWithCode(RAGException):
            def __init__(self, message, error_code=None):
                super().__init__(message)
                self.error_code = error_code
        
        error = ErrorWithCode("Test error", error_code="E001")
        assert str(error) == "Test error"
        assert error.error_code == "E001"