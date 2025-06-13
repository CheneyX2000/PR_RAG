# tests/e2e/test_end_to_end.py
"""
End-to-end tests for the complete RAG system.
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock
from uuid import uuid4
import json
import time

from httpx import AsyncClient
from src.rag_system.services.retriever import RetrievedChunk
from src.rag_system.services.generator import GenerationResponse


class TestE2EBasicFlow:
    """Test basic end-to-end flow"""
    
    @pytest.mark.asyncio
    async def test_complete_rag_flow(self, client: AsyncClient):
        """Test complete flow: ingest -> search -> generate"""
        # Step 1: Ingest a document
        doc_id = str(uuid4())
        
        with patch('src.rag_system.services.ingestion.ingestion_service.ingest_document') as mock_ingest:
            mock_ingest.return_value = doc_id
            
            ingest_response = await client.post(
                "/api/v1/ingest",
                json={
                    "title": "Introduction to RAG",
                    "content": "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval with generation.",
                    "metadata": {"topic": "AI"}
                }
            )
            
            assert ingest_response.status_code == 200
            assert ingest_response.json()["document_id"] == doc_id
        
        # Step 2: Search for the document
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            mock_search.return_value = [
                RetrievedChunk(
                    document_id=doc_id,
                    document_title="Introduction to RAG",
                    chunk_id="chunk1",
                    content="Retrieval-Augmented Generation (RAG) is a technique...",
                    chunk_index=0,
                    similarity_score=0.95
                )
            ]
            
            search_response = await client.post(
                "/api/v1/search",
                json={
                    "query": "What is RAG?",
                    "top_k": 5
                }
            )
            
            assert search_response.status_code == 200
            results = search_response.json()["results"]
            assert len(results) > 0
            assert results[0]["document_id"] == doc_id
        
        # Step 3: Generate answer using retrieved documents
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            with patch('src.rag_system.services.generator.generator_service.generate') as mock_generate:
                mock_search.return_value = [
                    RetrievedChunk(
                        document_id=doc_id,
                        document_title="Introduction to RAG",
                        chunk_id="chunk1",
                        content="Retrieval-Augmented Generation (RAG) is a technique...",
                        chunk_index=0,
                        similarity_score=0.95
                    )
                ]
                
                mock_generate.return_value = GenerationResponse(
                    text="RAG is a technique that combines information retrieval with text generation to provide accurate, contextual responses.",
                    model_name="gpt-4o-mini",
                    token_count=50
                )
                
                query_response = await client.post(
                    "/api/v1/query",
                    json={
                        "query": "What is RAG?",
                        "model": "gpt-4o-mini"
                    }
                )
                
                assert query_response.status_code == 200
                answer = query_response.json()
                assert "RAG is a technique" in answer["answer"]
                assert len(answer["sources"]) > 0
                assert answer["sources"][0]["document_id"] == doc_id


class TestE2EWithReranking:
    """Test end-to-end flow with reranking"""
    
    @pytest.mark.asyncio
    async def test_search_with_reranking(self, client: AsyncClient):
        """Test search with reranking improving results"""
        # Initial search results (before reranking)
        initial_chunks = [
            RetrievedChunk(
                document_id="doc1",
                document_title="Irrelevant Doc",
                chunk_id="chunk1",
                content="Some unrelated content",
                chunk_index=0,
                similarity_score=0.9  # High similarity but irrelevant
            ),
            RetrievedChunk(
                document_id="doc2",
                document_title="Relevant Doc",
                chunk_id="chunk2",
                content="This explains RAG in detail",
                chunk_index=0,
                similarity_score=0.85  # Lower similarity but more relevant
            )
        ]
        
        # After reranking
        reranked_chunks = [
            RetrievedChunk(
                document_id="doc2",
                document_title="Relevant Doc",
                chunk_id="chunk2",
                content="This explains RAG in detail",
                chunk_index=0,
                similarity_score=0.85,
                rerank_score=0.95  # High rerank score
            ),
            RetrievedChunk(
                document_id="doc1",
                document_title="Irrelevant Doc",
                chunk_id="chunk1",
                content="Some unrelated content",
                chunk_index=0,
                similarity_score=0.9,
                rerank_score=0.3  # Low rerank score
            )
        ]
        
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            # Return different results based on rerank parameter
            mock_search.side_effect = lambda **kwargs: reranked_chunks if kwargs.get('rerank') else initial_chunks
            
            # Search without reranking
            response1 = await client.post(
                "/api/v1/search",
                json={
                    "query": "Explain RAG",
                    "top_k": 2,
                    "rerank": False
                }
            )
            
            # Search with reranking
            response2 = await client.post(
                "/api/v1/search",
                json={
                    "query": "Explain RAG",
                    "top_k": 2,
                    "rerank": True,
                    "rerank_model": "ms-marco-MiniLM-L-6-v2"
                }
            )
            
            # Verify reranking improved results
            results1 = response1.json()["results"]
            results2 = response2.json()["results"]
            
            assert results1[0]["document_id"] == "doc1"  # Irrelevant doc first
            assert results2[0]["document_id"] == "doc2"  # Relevant doc first after reranking
            assert results2[0]["rerank_score"] > results2[1]["rerank_score"]


class TestE2EBatchOperations:
    """Test batch operations end-to-end"""
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_and_search(self, client: AsyncClient):
        """Test batch document ingestion and multi-doc search"""
        # Batch ingest documents
        doc_ids = [str(uuid4()) for _ in range(3)]
        
        with patch('src.rag_system.services.ingestion.ingestion_service.ingest_document') as mock_ingest:
            mock_ingest.side_effect = doc_ids
            
            batch_response = await client.post(
                "/api/v1/ingest/batch",
                json=[
                    {
                        "title": f"Document {i}",
                        "content": f"Content for document {i}"
                    }
                    for i in range(3)
                ]
            )
            
            assert batch_response.status_code == 200
            results = batch_response.json()
            assert len(results) == 3
            assert all(r["status"] == "success" for r in results)
        
        # Search across all documents
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            mock_search.return_value = [
                RetrievedChunk(
                    document_id=doc_ids[i],
                    document_title=f"Document {i}",
                    chunk_id=f"chunk{i}",
                    content=f"Content for document {i}",
                    chunk_index=0,
                    similarity_score=0.9 - i * 0.1
                )
                for i in range(3)
            ]
            
            search_response = await client.post(
                "/api/v1/search",
                json={
                    "query": "document content",
                    "top_k": 10
                }
            )
            
            assert search_response.status_code == 200
            results = search_response.json()["results"]
            assert len(results) == 3
            
            # Verify all documents were found
            found_doc_ids = {r["document_id"] for r in results}
            assert found_doc_ids == set(doc_ids)


class TestE2EStreamingResponse:
    """Test streaming response end-to-end"""
    
    @pytest.mark.asyncio
    async def test_streaming_query(self, client: AsyncClient):
        """Test streaming query response"""
        mock_chunks = [
            RetrievedChunk(
                document_id="doc1",
                document_title="Test Doc",
                chunk_id="chunk1",
                content="Test content",
                chunk_index=0,
                similarity_score=0.9
            )
        ]
        
        async def mock_stream():
            tokens = ["This ", "is ", "a ", "streaming ", "response."]
            for token in tokens:
                yield token
        
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            with patch('src.rag_system.services.generator.generator_service.stream_generate') as mock_stream_gen:
                mock_search.return_value = mock_chunks
                mock_stream_gen.return_value = mock_stream()
                
                response = await client.post(
                    "/api/v1/query/stream",
                    json={"query": "test streaming"},
                    headers={"Accept": "text/event-stream"}
                )
                
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                
                # Parse SSE response
                content = response.content.decode()
                lines = content.strip().split('\n')
                
                # Extract token events
                tokens = []
                for line in lines:
                    if line.startswith('data: ') and line != 'data: ':
                        data = json.loads(line[6:])
                        if data.get('type') == 'token':
                            tokens.append(data['content'])
                
                assert "".join(tokens) == "This is a streaming response."


class TestE2EErrorHandling:
    """Test error handling throughout the pipeline"""
    
    @pytest.mark.asyncio
    async def test_ingestion_error_handling(self, client: AsyncClient):
        """Test error handling during ingestion"""
        with patch('src.rag_system.services.ingestion.ingestion_service.ingest_document') as mock_ingest:
            mock_ingest.side_effect = Exception("Database connection failed")
            
            response = await client.post(
                "/api/v1/ingest",
                json={
                    "title": "Test",
                    "content": "Content"
                }
            )
            
            assert response.status_code == 500
            assert "Database connection failed" in response.text
    
    @pytest.mark.asyncio
    async def test_search_error_recovery(self, client: AsyncClient):
        """Test search error recovery"""
        # First search fails
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            mock_search.side_effect = Exception("Search index unavailable")
            
            response1 = await client.post(
                "/api/v1/search",
                json={"query": "test"}
            )
            
            assert response1.status_code == 500
        
        # Second search succeeds (simulating recovery)
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            mock_search.return_value = []
            
            response2 = await client.post(
                "/api/v1/search",
                json={"query": "test"}
            )
            
            assert response2.status_code == 200
    
    @pytest.mark.asyncio
    async def test_generation_fallback(self, client: AsyncClient):
        """Test generation with fallback when primary model fails"""
        mock_chunks = [
            RetrievedChunk(
                document_id="doc1",
                document_title="Test",
                chunk_id="chunk1",
                content="Content",
                chunk_index=0,
                similarity_score=0.9
            )
        ]
        
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            mock_search.return_value = mock_chunks
            
            # First attempt fails
            with patch('src.rag_system.services.generator.generator_service.generate') as mock_gen:
                mock_gen.side_effect = Exception("Model unavailable")
                
                response = await client.post(
                    "/api/v1/query",
                    json={"query": "test", "model": "gpt-4o"}
                )
                
                assert response.status_code == 500


class TestE2EPerformance:
    """Test system performance end-to-end"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client: AsyncClient):
        """Test handling concurrent requests"""
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            mock_search.return_value = []
            
            # Send multiple concurrent requests
            tasks = []
            for i in range(10):
                task = client.post(
                    "/api/v1/search",
                    json={"query": f"query {i}"}
                )
                tasks.append(task)
            
            # Wait for all requests
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(r.status_code == 200 for r in responses)
    
    @pytest.mark.asyncio
    async def test_response_time_monitoring(self, client: AsyncClient):
        """Test that response times are monitored"""
        with patch('src.rag_system.services.retriever.retriever.search') as mock_search:
            mock_search.return_value = []
            
            start_time = time.time()
            response = await client.post(
                "/api/v1/search",
                json={"query": "test"}
            )
            end_time = time.time()
            
            assert response.status_code == 200
            
            # Check response time header
            assert "X-Response-Time" in response.headers
            response_time = float(response.headers["X-Response-Time"])
            assert 0 < response_time < (end_time - start_time) + 0.1


class TestE2EHealthChecks:
    """Test health check integration"""
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, client: AsyncClient):
        """Test comprehensive health monitoring"""
        with patch('src.rag_system.api.routes.check_database_health', return_value=True):
            with patch('src.rag_system.api.routes.check_redis_health', return_value=True):
                with patch('src.rag_system.utils.circuit_breaker.CircuitBreaker.get_all_stats') as mock_stats:
                    mock_stats.return_value = {
                        "openai": {"state": "closed", "total_calls": 100, "failed_calls": 2},
                        "database": {"state": "closed", "total_calls": 500, "failed_calls": 5},
                        "redis": {"state": "closed", "total_calls": 200, "failed_calls": 1}
                    }
                    
                    # Basic health check
                    response1 = await client.get("/api/v1/health")
                    assert response1.status_code == 200
                    assert response1.json()["status"] == "healthy"
                    
                    # Detailed health check
                    response2 = await client.get("/api/v1/health/detailed")
                    assert response2.status_code == 200
                    
                    detailed = response2.json()
                    assert detailed["status"] == "healthy"
                    assert detailed["components"]["database"]["status"] == "healthy"
                    assert detailed["components"]["redis"]["status"] == "healthy"
                    assert detailed["circuit_breakers"]["total"] == 3
                    assert detailed["circuit_breakers"]["open"] == 0