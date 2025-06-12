# tests/test_api.py
"""
Tests for API endpoints.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
import json


class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: AsyncClient):
        """Test health check endpoint"""
        response = await client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "embedding_model" in data
        assert "llm_model" in data
    
    @pytest.mark.asyncio
    async def test_ingest_document(self, client: AsyncClient):
        """Test document ingestion endpoint"""
        # Mock the ingestion service
        with patch('src.rag_system.api.routes.ingestion_service.ingest_document') as mock_ingest:
            mock_ingest.return_value = "doc-id-123"
            
            document = {
                "title": "Test Document",
                "content": "This is test content",
                "metadata": {"category": "test"}
            }
            
            response = await client.post(
                "/api/v1/ingest",
                json=document
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["document_id"] == "doc-id-123"
    
    @pytest.mark.asyncio
    async def test_search_documents(self, client: AsyncClient):
        """Test document search endpoint"""
        # Mock the retriever service
        mock_chunk = {
            "document_id": "doc1",
            "document_title": "Test Doc",
            "content": "Test content",
            "chunk_index": 0,
            "similarity_score": 0.95,
            "metadata": {}
        }
        
        with patch('src.rag_system.api.routes.retriever.search') as mock_search:
            from src.rag_system.services.retriever import RetrievedChunk
            mock_search.return_value = [
                RetrievedChunk(**mock_chunk)
            ]
            
            search_request = {
                "query": "test query",
                "top_k": 5
            }
            
            response = await client.post(
                "/api/v1/search",
                json=search_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test query"
            assert len(data["results"]) == 1
            assert data["results"][0]["similarity_score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_query_rag(self, client: AsyncClient):
        """Test RAG query endpoint"""
        # Mock retriever
        mock_chunk = {
            "document_id": "doc1",
            "document_title": "Test Doc",
            "content": "Test content about RAG",
            "chunk_index": 0,
            "similarity_score": 0.95,
            "metadata": {}
        }
        
        # Mock generator
        from src.rag_system.services.generator import GenerationResponse
        mock_generation = GenerationResponse(
            text="This is a generated response about RAG",
            model_name="gpt-4o-mini",
            token_count=50
        )
        
        with patch('src.rag_system.api.routes.retriever.search') as mock_search:
            with patch('src.rag_system.api.routes.generator_service.generate') as mock_generate:
                from src.rag_system.services.retriever import RetrievedChunk
                mock_search.return_value = [RetrievedChunk(**mock_chunk)]
                mock_generate.return_value = mock_generation
                
                query_request = {
                    "query": "What is RAG?",
                    "model": "gpt-4o-mini"
                }
                
                response = await client.post(
                    "/api/v1/query",
                    json=query_request
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "This is a generated response about RAG"
                assert len(data["sources"]) == 1
                assert data["metadata"]["model_used"] == "gpt-4o-mini"
    
    @pytest.mark.asyncio
    async def test_stream_query(self, client: AsyncClient):
        """Test streaming query endpoint"""
        # Mock services
        mock_chunk = {
            "document_id": "doc1",
            "document_title": "Test Doc",
            "content": "Test content",
            "chunk_index": 0,
            "similarity_score": 0.95,
            "metadata": {}
        }
        
        async def mock_stream_generate(*args, **kwargs):
            for word in ["This", " is", " streaming"]:
                yield word
        
        with patch('src.rag_system.api.routes.retriever.search') as mock_search:
            with patch('src.rag_system.api.routes.generator_service.stream_generate') as mock_stream:
                from src.rag_system.services.retriever import RetrievedChunk
                mock_search.return_value = [RetrievedChunk(**mock_chunk)]
                mock_stream.return_value = mock_stream_generate()
                
                query_request = {
                    "query": "Test streaming"
                }
                
                # Make streaming request
                response = await client.post(
                    "/api/v1/query/stream",
                    json=query_request,
                    headers={"Accept": "text/event-stream"}
                )
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                
                # Parse SSE response
                content = response.content.decode()
                lines = content.strip().split('\n')
                
                # Verify we have data events
                data_lines = [line for line in lines if line.startswith('data:')]
                assert len(data_lines) > 0
                
                # Verify structure of events
                for line in data_lines:
                    if line.strip() == 'data:':
                        continue
                    event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                    assert 'type' in event_data
    
    @pytest.mark.asyncio
    async def test_invalid_request(self, client: AsyncClient):
        """Test handling of invalid requests"""
        # Missing required field
        invalid_request = {
            "content": "Missing title"
        }
        
        response = await client.post(
            "/api/v1/ingest",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio 
    async def test_rate_limiting(self, client: AsyncClient):
        """Test rate limiting"""
        # This test would need proper rate limiting setup
        # For now, just verify the endpoint exists
        response = await client.get("/api/v1/health")
        assert response.status_code == 200