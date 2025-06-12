# tests/conftest.py
"""
Pytest configuration and fixtures for RAG system tests.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
import os
from unittest.mock import Mock, AsyncMock

# Set test environment
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test_ragdb"
os.environ["OPENAI_API_KEY"] = "test-key"

from src.rag_system.db.pgvector import PgVectorDB
from src.rag_system.services.embeddings import EmbeddingService
from src.rag_system.services.retriever import RetrieverService
from src.rag_system.services.ingestion import DocumentIngestionService
from src.rag_system.services.generator import GeneratorService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_db() -> AsyncGenerator[PgVectorDB, None]:
    """Create a test database instance."""
    db = PgVectorDB("postgresql://test:test@localhost:5432/test_ragdb")
    await db.initialize()
    
    yield db
    
    # Cleanup
    await db.close()


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Create a mock embedding service."""
    service = Mock(spec=EmbeddingService)
    
    # Mock embed_text to return a fixed embedding
    async def mock_embed_text(text: str):
        # Return a simple embedding based on text length
        return [0.1] * 1536  # Standard embedding size
    
    async def mock_embed_texts(texts: list):
        return [[0.1] * 1536 for _ in texts]
    
    service.embed_text = AsyncMock(side_effect=mock_embed_text)
    service.embed_texts = AsyncMock(side_effect=mock_embed_texts)
    service.current_model_name = "test-embedding-model"
    
    return service


@pytest.fixture
def mock_generator_service() -> GeneratorService:
    """Create a mock generator service."""
    service = Mock(spec=GeneratorService)
    
    async def mock_generate(query, context_chunks, **kwargs):
        from src.rag_system.services.generator import GenerationResponse
        return GenerationResponse(
            text=f"Generated response for: {query}",
            model_name="test-model",
            token_count=100
        )
    
    service.generate = AsyncMock(side_effect=mock_generate)
    return service


@pytest.fixture
async def retriever_service(test_db) -> RetrieverService:
    """Create a retriever service instance."""
    return RetrieverService()


@pytest.fixture
async def ingestion_service() -> DocumentIngestionService:
    """Create an ingestion service instance."""
    return DocumentIngestionService()


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "title": "Test Document 1",
            "content": "This is a test document about RAG systems and vector databases.",
            "metadata": {"category": "test"}
        },
        {
            "title": "Test Document 2", 
            "content": "PgVector is a PostgreSQL extension for vector similarity search.",
            "metadata": {"category": "test"}
        }
    ]


@pytest.fixture
async def app():
    """Create FastAPI test app."""
    from src.rag_system.main import app
    return app


@pytest.fixture
async def client(app):
    """Create async test client."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Mock OpenAI responses
@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """Mock OpenAI API calls."""
    async def mock_completion(**kwargs):
        class MockChoice:
            class MockMessage:
                content = "Mocked response"
            message = MockMessage()
        
        class MockUsage:
            total_tokens = 100
        
        class MockResponse:
            choices = [MockChoice()]
            usage = MockUsage()
        
        return MockResponse()
    
    # Mock for embeddings
    async def mock_embeddings(**kwargs):
        class MockEmbedding:
            embedding = [0.1] * 1536
        
        class MockResponse:
            data = [MockEmbedding() for _ in kwargs.get("input", [])]
        
        return MockResponse()
    
    # Apply mocks
    import openai
    monkeypatch.setattr("openai.AsyncOpenAI.embeddings.create", mock_embeddings)
    monkeypatch.setattr("litellm.acompletion", mock_completion)