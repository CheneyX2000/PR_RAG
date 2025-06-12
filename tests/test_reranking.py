# tests/test_reranking.py
"""
Test suite for the reranking functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.rag_system.services.retriever import (
    RetrieverService,
    RetrievedChunk,
    RerankingModel,
    RerankingConfig
)


class TestRerankingModel:
    """Test cases for RerankingModel"""
    
    @pytest.fixture
    def reranking_config(self):
        """Create a test reranking configuration"""
        return RerankingConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=2,
            normalize_scores=True
        )
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing"""
        return [
            RetrievedChunk(
                document_id="doc1",
                document_title="Test Doc 1",
                chunk_id="chunk1",
                content="This is about reranking in information retrieval",
                chunk_index=0,
                similarity_score=0.85
            ),
            RetrievedChunk(
                document_id="doc2",
                document_title="Test Doc 2",
                chunk_id="chunk2",
                content="Vector search is important for RAG systems",
                chunk_index=0,
                similarity_score=0.82
            ),
            RetrievedChunk(
                document_id="doc3",
                document_title="Test Doc 3",
                chunk_id="chunk3",
                content="Cross-encoders improve search quality significantly",
                chunk_index=0,
                similarity_score=0.80
            )
        ]
    
    @pytest.mark.asyncio
    async def test_reranking_model_initialization(self, reranking_config):
        """Test reranking model initialization"""
        model = RerankingModel(reranking_config)
        
        assert model.config == reranking_config
        assert model._model is None  # Lazy loading
        assert model._circuit_breaker is not None
    
    @pytest.mark.asyncio
    async def test_reranking_basic(self, reranking_config, sample_chunks):
        """Test basic reranking functionality"""
        model = RerankingModel(reranking_config)
        
        # Mock the cross-encoder predict method
        with patch.object(model, 'model') as mock_model:
            # Return different scores to show reordering
            mock_model.predict.return_value = np.array([0.7, 0.9, 0.95])
            
            query = "What is reranking?"
            reranked = await model.rerank(query, sample_chunks)
            
            # Check that chunks were reordered
            assert len(reranked) == 3
            assert reranked[0].chunk_id == "chunk3"  # Highest rerank score
            assert reranked[1].chunk_id == "chunk2"
            assert reranked[2].chunk_id == "chunk1"  # Lowest rerank score
            
            # Check rerank scores were assigned
            assert all(chunk.rerank_score is not None for chunk in reranked)
    
    @pytest.mark.asyncio
    async def test_reranking_with_top_k(self, reranking_config, sample_chunks):
        """Test reranking with top_k limit"""
        model = RerankingModel(reranking_config)
        
        with patch.object(model, 'model') as mock_model:
            mock_model.predict.return_value = np.array([0.7, 0.9, 0.95])
            
            reranked = await model.rerank("test query", sample_chunks, top_k=2)
            
            assert len(reranked) == 2
            assert reranked[0].chunk_id == "chunk3"
            assert reranked[1].chunk_id == "chunk2"
    
    @pytest.mark.asyncio
    async def test_reranking_normalization(self, reranking_config, sample_chunks):
        """Test score normalization"""
        model = RerankingModel(reranking_config)
        
        with patch.object(model, 'model') as mock_model:
            # Return scores with different ranges
            mock_model.predict.return_value = np.array([-2.0, 0.0, 5.0])
            
            reranked = await model.rerank("test query", sample_chunks)
            
            # Check scores are normalized to [0, 1]
            scores = [chunk.rerank_score for chunk in reranked]
            assert min(scores) >= 0.0
            assert max(scores) <= 1.0
            assert scores[0] == 1.0  # Highest original score
            assert scores[-1] == 0.0  # Lowest original score
    
    @pytest.mark.asyncio
    async def test_reranking_empty_input(self, reranking_config):
        """Test reranking with empty input"""
        model = RerankingModel(reranking_config)
        
        reranked = await model.rerank("test query", [])
        assert reranked == []
    
    @pytest.mark.asyncio
    async def test_reranking_error_handling(self, reranking_config, sample_chunks):
        """Test error handling in reranking"""
        model = RerankingModel(reranking_config)
        
        with patch.object(model, 'model') as mock_model:
            mock_model.predict.side_effect = Exception("Model error")
            
            # Should return original chunks on error
            reranked = await model.rerank("test query", sample_chunks)
            
            assert len(reranked) == len(sample_chunks)
            # Original order preserved
            assert [c.chunk_id for c in reranked] == [c.chunk_id for c in sample_chunks]


class TestRetrieverServiceReranking:
    """Test cases for RetrieverService reranking integration"""
    
    @pytest.fixture
    def retriever(self):
        """Create a retriever service instance"""
        return RetrieverService()
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service"""
        with patch('src.rag_system.services.retriever.embedding_service') as mock:
            mock.current_model_name = "test-model"
            mock.get_model_dimension.return_value = 1536
            mock.embed_text.return_value = ([0.1] * 1536, 1536)
            yield mock
    
    @pytest.fixture
    def mock_db(self):
        """Mock database"""
        with patch('src.rag_system.services.retriever.db') as mock:
            yield mock
    
    @pytest.mark.asyncio
    async def test_search_with_reranking_disabled(
        self, retriever, mock_embedding_service, mock_db
    ):
        """Test search with reranking disabled"""
        # Mock db results
        mock_db.similarity_search.return_value = [
            {
                'document_id': 'doc1',
                'title': 'Test Doc',
                'chunk_id': 'chunk1',
                'content': 'Test content',
                'chunk_index': 0,
                'similarity': 0.9,
                'metadata': {}
            }
        ]
        
        results = await retriever.search(
            query="test query",
            rerank=False
        )
        
        assert len(results) == 1
        assert results[0].rerank_score is None
    
    @pytest.mark.asyncio
    async def test_search_with_reranking_enabled(
        self, retriever, mock_embedding_service, mock_db
    ):
        """Test search with reranking enabled"""
        # Mock db results
        mock_db.similarity_search.return_value = [
            {
                'document_id': f'doc{i}',
                'title': f'Test Doc {i}',
                'chunk_id': f'chunk{i}',
                'content': f'Test content {i}',
                'chunk_index': 0,
                'similarity': 0.9 - i * 0.1,
                'metadata': {}
            }
            for i in range(3)
        ]
        
        # Mock reranking
        with patch.object(retriever, '_rerank_results') as mock_rerank:
            mock_rerank.return_value = [
                RetrievedChunk(
                    document_id='doc2',
                    document_title='Test Doc 2',
                    chunk_id='chunk2',
                    content='Test content 2',
                    chunk_index=0,
                    similarity_score=0.7,
                    rerank_score=0.95
                )
            ]
            
            results = await retriever.search(
                query="test query",
                rerank=True,
                top_k=1
            )
            
            assert len(results) == 1
            assert results[0].rerank_score == 0.95
            mock_rerank.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_with_specific_rerank_model(
        self, retriever, mock_embedding_service, mock_db
    ):
        """Test search with specific reranking model"""
        mock_db.similarity_search.return_value = [{
            'document_id': 'doc1',
            'title': 'Test',
            'chunk_id': 'chunk1',
            'content': 'content',
            'chunk_index': 0,
            'similarity': 0.9,
            'metadata': {}
        }]
        
        with patch.object(retriever, '_rerank_results') as mock_rerank:
            mock_rerank.return_value = []
            
            await retriever.search(
                query="test",
                rerank=True,
                rerank_model="ms-marco-MiniLM-L-12-v2"
            )
            
            # Check that the model name was passed
            mock_rerank.assert_called_once()
            call_args = mock_rerank.call_args
            assert call_args[1]['model_name'] == "ms-marco-MiniLM-L-12-v2"
    
    def test_enable_reranking(self, retriever):
        """Test enabling/disabling reranking"""
        # Initially disabled
        assert retriever.reranking_enabled == False
        
        # Enable with default model
        retriever.enable_reranking(True)
        assert retriever.reranking_enabled == True
        assert retriever._default_reranking_model == "ms-marco-MiniLM-L-6-v2"
        
        # Enable with specific model
        retriever.enable_reranking(True, model="ms-marco-MiniLM-L-12-v2")
        assert retriever._default_reranking_model == "ms-marco-MiniLM-L-12-v2"
        
        # Disable
        retriever.enable_reranking(False)
        assert retriever.reranking_enabled == False
    
    def test_get_reranking_models(self, retriever):
        """Test getting available reranking models"""
        models = retriever.get_reranking_models()
        
        assert isinstance(models, dict)
        assert "ms-marco-MiniLM-L-6-v2" in models
        assert "ms-marco-MiniLM-L-12-v2" in models
        assert "ms-marco-TinyBERT-L-2-v2" in models
        
        # Check model info structure
        model_info = models["ms-marco-MiniLM-L-6-v2"]
        assert "model_name" in model_info
        assert "batch_size" in model_info
        assert "is_default" in model_info
        assert model_info["is_default"] == True  # Default model
    
    @pytest.mark.asyncio
    async def test_search_multi_model_with_reranking(
        self, retriever, mock_embedding_service, mock_db
    ):
        """Test multi-model search with reranking"""
        # Mock different results for different models
        mock_db.similarity_search.return_value = [{
            'document_id': 'doc1',
            'title': 'Test',
            'chunk_id': 'chunk1',
            'content': 'content',
            'chunk_index': 0,
            'similarity': 0.9,
            'metadata': {}
        }]
        
        with patch.object(retriever, '_rerank_results') as mock_rerank:
            mock_rerank.return_value = []
            
            await retriever.search_multi_model(
                query="test",
                model_names=["model1", "model2"],
                rerank=True
            )
            
            # Should call rerank once for aggregated results
            mock_rerank.assert_called_once()


class TestRerankingIntegration:
    """Integration tests for reranking"""
    
    @pytest.mark.asyncio
    async def test_reranking_improves_relevance(self):
        """Test that reranking actually improves result relevance"""
        retriever = RetrieverService()
        
        # Create test chunks with misleading similarity scores
        chunks = [
            RetrievedChunk(
                document_id="doc1",
                document_title="Unrelated Doc",
                chunk_id="chunk1",
                content="This document is about cooking recipes and has nothing to do with the query",
                chunk_index=0,
                similarity_score=0.9  # High similarity but irrelevant
            ),
            RetrievedChunk(
                document_id="doc2",
                document_title="Relevant Doc",
                chunk_id="chunk2",
                content="Cross-encoder models for reranking significantly improve search quality in RAG",
                chunk_index=0,
                similarity_score=0.7  # Lower similarity but more relevant
            )
        ]
        
        # Mock the cross-encoder to give better scores to relevant content
        model = RerankingModel(retriever.RERANKING_MODELS["ms-marco-MiniLM-L-6-v2"])
        
        with patch.object(model, 'model') as mock_model:
            # Relevant document gets higher score
            mock_model.predict.return_value = np.array([0.2, 0.9])
            
            query = "How do cross-encoders improve RAG systems?"
            reranked = await model.rerank(query, chunks)
            
            # Relevant document should be first after reranking
            assert reranked[0].chunk_id == "chunk2"
            assert reranked[0].rerank_score > reranked[1].rerank_score