# tests/integration/test_rag_pipeline.py
"""
Comprehensive integration tests for the complete RAG pipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
import json

from src.rag_system.services.ingestion import ingestion_service
from src.rag_system.services.retriever import retriever
from src.rag_system.services.generator import generator_service
from src.rag_system.services.embeddings import embedding_service
from src.rag_system.db.pgvector import db


class TestRAGPipeline:
    """End-to-end tests for the RAG pipeline"""
    
    @pytest.fixture
    async def setup_pipeline(self):
        """Set up the pipeline with mocked dependencies"""
        # Mock database
        with patch('src.rag_system.db.pgvector.db') as mock_db:
            mock_db.get_session.return_value.__aenter__.return_value = AsyncMock()
            mock_db.insert_embedding = AsyncMock(return_value=uuid4())
            mock_db.similarity_search = AsyncMock(return_value=[])
            mock_db.ensure_embedding_model = AsyncMock(return_value=uuid4())
            mock_db._dimension_cache = {"test-model": 1536}
            
            # Mock embedding service
            with patch('src.rag_system.services.embeddings.embedding_service') as mock_emb:
                mock_emb.current_model_name = "test-model"
                mock_emb.embed_text = AsyncMock(return_value=([0.1] * 1536, 1536))
                mock_emb.embed_texts = AsyncMock(return_value=([[0.1] * 1536], 1536))
                mock_emb.get_model_dimension = Mock(return_value=1536)
                
                yield mock_db, mock_emb
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self, setup_pipeline):
        """Test complete RAG pipeline from ingestion to generation"""
        mock_db, mock_emb = setup_pipeline
        
        # Step 1: Ingest a document
        doc_content = "RAG systems combine retrieval with generation for better results."
        
        with patch('src.rag_system.utils.chunking.TextChunker.split_text') as mock_chunk:
            from src.rag_system.utils.chunking import TextChunk
            mock_chunk.return_value = [
                TextChunk(
                    content=doc_content,
                    start_char=0,
                    end_char=len(doc_content),
                    chunk_index=0
                )
            ]
            
            # Mock session operations
            mock_session = mock_db.get_session.return_value.__aenter__.return_value
            doc_id = uuid4()
            chunk_id = uuid4()
            
            def mock_add(obj):
                if hasattr(obj, 'title'):  # Document
                    obj.id = doc_id
                else:  # Chunk
                    obj.id = chunk_id
            
            mock_session.add.side_effect = mock_add
            
            # Ingest document
            result_id = await ingestion_service.ingest_document(
                title="RAG Introduction",
                content=doc_content
            )
            
            assert result_id == doc_id
        
        # Step 2: Search for relevant documents
        query = "How do RAG systems work?"
        
        mock_db.similarity_search.return_value = [{
            'document_id': str(doc_id),
            'title': 'RAG Introduction',
            'chunk_id': str(chunk_id),
            'content': doc_content,
            'chunk_index': 0,
            'similarity': 0.92,
            'metadata': {}
        }]
        
        search_results = await retriever.search(query, top_k=5)
        
        assert len(search_results) == 1
        assert search_results[0].similarity_score == 0.92
        assert search_results[0].content == doc_content
        
        # Step 3: Generate response
        with patch('litellm.acompletion') as mock_llm:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="RAG systems work by..."))]
            mock_response.usage = Mock(total_tokens=100)
            mock_llm.return_value = mock_response
            
            response = await generator_service.generate(
                query=query,
                context_chunks=search_results,
                model_name="gpt-4o-mini"
            )
            
            assert response.text == "RAG systems work by..."
            assert response.token_count == 100
            assert response.metadata["context_chunks_used"] == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_documents(self, setup_pipeline):
        """Test pipeline with multiple documents and chunks"""
        mock_db, mock_emb = setup_pipeline
        
        # Prepare multiple documents
        documents = [
            {
                "title": "Doc 1",
                "content": "First document about RAG systems."
            },
            {
                "title": "Doc 2", 
                "content": "Second document about vector databases."
            },
            {
                "title": "Doc 3",
                "content": "Third document about embeddings."
            }
        ]
        
        # Mock embedding generation for multiple texts
        mock_emb.embed_texts.return_value = (
            [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536],
            1536
        )
        
        # Ingest documents
        with patch('src.rag_system.utils.chunking.TextChunker.split_text') as mock_chunk:
            from src.rag_system.utils.chunking import TextChunk
            
            doc_ids = []
            for i, doc in enumerate(documents):
                mock_chunk.return_value = [
                    TextChunk(
                        content=doc["content"],
                        start_char=0,
                        end_char=len(doc["content"]),
                        chunk_index=0
                    )
                ]
                
                doc_id = uuid4()
                mock_session = mock_db.get_session.return_value.__aenter__.return_value
                mock_session.add.side_effect = lambda obj: setattr(obj, 'id', doc_id)
                
                result_id = await ingestion_service.ingest_document(**doc)
                doc_ids.append(result_id)
        
        # Search across all documents
        mock_db.similarity_search.return_value = [
            {
                'document_id': str(doc_ids[0]),
                'title': 'Doc 1',
                'chunk_id': str(uuid4()),
                'content': documents[0]["content"],
                'chunk_index': 0,
                'similarity': 0.95,
                'metadata': {}
            },
            {
                'document_id': str(doc_ids[2]),
                'title': 'Doc 3',
                'chunk_id': str(uuid4()),
                'content': documents[2]["content"],
                'chunk_index': 0,
                'similarity': 0.88,
                'metadata': {}
            }
        ]
        
        results = await retriever.search("RAG and embeddings", top_k=2)
        
        assert len(results) == 2
        assert results[0].document_title == "Doc 1"
        assert results[1].document_title == "Doc 3"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_reranking(self, setup_pipeline):
        """Test pipeline with reranking enabled"""
        mock_db, mock_emb = setup_pipeline
        
        # Enable reranking
        retriever.enable_reranking(True)
        
        # Mock initial search results
        initial_results = [
            {
                'document_id': 'doc1',
                'title': 'Low relevance doc',
                'chunk_id': 'chunk1',
                'content': 'Some content about unrelated topics',
                'chunk_index': 0,
                'similarity': 0.9,  # High vector similarity
                'metadata': {}
            },
            {
                'document_id': 'doc2',
                'title': 'High relevance doc',
                'chunk_id': 'chunk2',
                'content': 'Detailed explanation of RAG systems',
                'chunk_index': 0,
                'similarity': 0.85,  # Lower vector similarity
                'metadata': {}
            }
        ]
        
        mock_db.similarity_search.return_value = initial_results
        
        # Mock reranking to reverse the order
        with patch('src.rag_system.services.retriever.RerankingModel') as mock_rerank_class:
            mock_reranker = Mock()
            mock_reranker.rerank = AsyncMock()
            
            # Reranker puts doc2 first
            async def mock_rerank(query, chunks, top_k=None):
                chunks[0].rerank_score = 0.6  # Low rerank score
                chunks[1].rerank_score = 0.95  # High rerank score
                return sorted(chunks, key=lambda x: x.rerank_score, reverse=True)[:top_k]
            
            mock_reranker.rerank.side_effect = mock_rerank
            mock_rerank_class.return_value = mock_reranker
            
            results = await retriever.search("Explain RAG systems", top_k=2, rerank=True)
            
            # Verify reranking changed the order
            assert len(results) == 2
            assert results[0].document_title == "High relevance doc"
            assert results[0].rerank_score == 0.95
            assert results[1].document_title == "Low relevance doc"
            assert results[1].rerank_score == 0.6
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, setup_pipeline):
        """Test pipeline error handling and recovery"""
        mock_db, mock_emb = setup_pipeline
        
        # Test ingestion error recovery
        with patch('src.rag_system.utils.chunking.TextChunker.split_text') as mock_chunk:
            mock_chunk.side_effect = Exception("Chunking failed")
            
            with pytest.raises(Exception):
                await ingestion_service.ingest_document(
                    title="Test",
                    content="Content"
                )
        
        # Test search with database error
        mock_db.similarity_search.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            await retriever.search("test query")
        
        # Reset for generation test
        mock_db.similarity_search.side_effect = None
        mock_db.similarity_search.return_value = []
        
        # Test generation with no context
        with patch('litellm.acompletion') as mock_llm:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="No context available"))]
            mock_response.usage = Mock(total_tokens=50)
            mock_llm.return_value = mock_response
            
            response = await generator_service.generate(
                query="test",
                context_chunks=[],
                model_name="gpt-4o-mini"
            )
            
            assert response.text == "No context available"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_caching(self, setup_pipeline):
        """Test pipeline with caching enabled"""
        mock_db, mock_emb = setup_pipeline
        
        # Mock Redis cache
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        
        from src.rag_system.services.cache import RAGCache
        cache = RAGCache(mock_redis)
        
        # First search - cache miss
        mock_db.similarity_search.return_value = [{
            'document_id': 'doc1',
            'title': 'Cached Doc',
            'chunk_id': 'chunk1',
            'content': 'Content',
            'chunk_index': 0,
            'similarity': 0.9,
            'metadata': {}
        }]
        
        # Simulate cache integration
        cache_key = "search:test query:5:0.0"
        cached_result = await cache.get_or_compute(
            cache_key,
            lambda: retriever.search("test query", top_k=5)
        )
        
        # Verify cache was populated
        mock_redis.setex.assert_called_once()
        
        # Second search - cache hit
        mock_redis.get.return_value = json.dumps([{
            "document_id": "doc1",
            "document_title": "Cached Doc",
            "content": "Content",
            "chunk_index": 0,
            "similarity_score": 0.9
        }])
        
        cached_result2 = await cache.get_or_compute(
            cache_key,
            lambda: retriever.search("test query", top_k=5)
        )
        
        # Verify result came from cache
        assert mock_db.similarity_search.call_count == 1  # Not called again


class TestRAGPipelineStreaming:
    """Test streaming functionality in the RAG pipeline"""
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self):
        """Test streaming response generation"""
        # Mock search results
        from src.rag_system.services.retriever import RetrievedChunk
        chunks = [
            RetrievedChunk(
                document_id="doc1",
                document_title="Test",
                chunk_id="chunk1",
                content="Test content",
                chunk_index=0,
                similarity_score=0.9
            )
        ]
        
        # Mock streaming response
        async def mock_stream():
            tokens = ["This ", "is ", "a ", "streaming ", "response."]
            for token in tokens:
                chunk = Mock()
                chunk.choices = [Mock(delta=Mock(content=token))]
                yield chunk
        
        with patch('litellm.acompletion', return_value=mock_stream()):
            collected = []
            async for token in generator_service.stream_generate(
                query="Test",
                context_chunks=chunks,
                model_name="gpt-4o-mini"
            ):
                collected.append(token)
            
            assert collected == ["This ", "is ", "a ", "streaming ", "response."]
            assert "".join(collected) == "This is a streaming response."