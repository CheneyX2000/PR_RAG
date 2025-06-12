# tests/test_retrieval.py
"""
Tests for the retrieval functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.rag_system.services.retriever import RetrieverService, RetrievedChunk


class TestRetrieverService:
    """Test cases for RetrieverService"""
    
    @pytest.mark.asyncio
    async def test_search_basic(self, retriever_service, mock_embedding_service):
        """Test basic search functionality"""
        # Mock the embedding service
        with patch('src.rag_system.services.retriever.embedding_service', mock_embedding_service):
            # Mock the database search
            mock_results = [
                {
                    'document_id': 'doc1',
                    'title': 'Test Document',
                    'chunk_id': 'chunk1',
                    'content': 'Test content',
                    'chunk_index': 0,
                    'similarity': 0.95,
                    'metadata': {}
                }
            ]
            
            with patch('src.rag_system.services.retriever.db.similarity_search', return_value=mock_results):
                results = await retriever_service.search(
                    query="test query",
                    top_k=5
                )
                
                assert len(results) == 1
                assert isinstance(results[0], RetrievedChunk)
                assert results[0].document_title == 'Test Document'
                assert results[0].similarity_score == 0.95
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, retriever_service, mock_embedding_service):
        """Test search with metadata filters"""
        with patch('src.rag_system.services.retriever.embedding_service', mock_embedding_service):
            with patch('src.rag_system.services.retriever.db.similarity_search') as mock_search:
                mock_search.return_value = []
                
                await retriever_service.search(
                    query="test query",
                    filters={"category": "test"}
                )
                
                # Verify filters were passed to database
                mock_search.assert_called_once()
                call_args = mock_search.call_args
                assert call_args[1]['filters'] == {"category": "test"}
    
    @pytest.mark.asyncio
    async def test_search_empty_results(self, retriever_service, mock_embedding_service):
        """Test search with no results"""
        with patch('src.rag_system.services.retriever.embedding_service', mock_embedding_service):
            with patch('src.rag_system.services.retriever.db.similarity_search', return_value=[]):
                results = await retriever_service.search(
                    query="non-existent query"
                )
                
                assert results == []
    
    @pytest.mark.asyncio
    async def test_search_error_handling(self, retriever_service, mock_embedding_service):
        """Test error handling in search"""
        with patch('src.rag_system.services.retriever.embedding_service', mock_embedding_service):
            with patch('src.rag_system.services.retriever.db.similarity_search', 
                      side_effect=Exception("Database error")):
                
                with pytest.raises(Exception) as exc_info:
                    await retriever_service.search("test query")
                
                assert "Database error" in str(exc_info.value)