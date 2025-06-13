# tests/test_ingestion.py
"""
Comprehensive tests for the document ingestion service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime

from src.rag_system.services.ingestion import DocumentIngestionService
from src.rag_system.db.models import Document, DocumentChunk, ChunkEmbedding
from src.rag_system.utils.chunking import TextChunk
from src.rag_system.utils.exceptions import IngestionError


class TestDocumentIngestionService:
    """Test cases for document ingestion service"""
    
    @pytest.fixture
    def ingestion_service(self):
        """Create ingestion service instance"""
        return DocumentIngestionService()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample text chunks"""
        return [
            TextChunk(
                content="First chunk of text",
                start_char=0,
                end_char=19,
                chunk_index=0
            ),
            TextChunk(
                content="Second chunk of text",
                start_char=20,
                end_char=40,
                chunk_index=1
            )
        ]
    
    @pytest.fixture
    def mock_session(self):
        """Create mock database session"""
        session = AsyncMock()
        session.add = Mock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session
    
    @pytest.mark.asyncio
    async def test_ingest_document_success(self, ingestion_service, sample_chunks, mock_session):
        """Test successful document ingestion"""
        # Mock dependencies
        doc_id = uuid4()
        chunk_ids = [uuid4(), uuid4()]
        
        with patch.object(ingestion_service.chunker, 'split_text', return_value=sample_chunks):
            with patch('src.rag_system.services.ingestion.embedding_service') as mock_emb:
                mock_emb.current_model_name = "text-embedding-ada-002"
                mock_emb.embed_texts.return_value = ([[0.1] * 1536, [0.2] * 1536], 1536)
                
                with patch('src.rag_system.services.ingestion.db.get_session') as mock_db:
                    mock_db.return_value.__aenter__.return_value = mock_session
                    
                    with patch('src.rag_system.services.ingestion.db.insert_embedding') as mock_insert:
                        mock_insert.return_value = uuid4()
                        
                        # Mock document and chunk creation
                        def mock_add(obj):
                            if isinstance(obj, Document):
                                obj.id = doc_id
                            elif isinstance(obj, DocumentChunk):
                                obj.id = chunk_ids.pop(0) if chunk_ids else uuid4()
                        
                        mock_session.add.side_effect = mock_add
                        
                        # Test ingestion
                        result_id = await ingestion_service.ingest_document(
                            title="Test Document",
                            content="First chunk of text Second chunk of text",
                            source_url="https://example.com",
                            document_type="article",
                            metadata={"author": "Test Author"}
                        )
                        
                        assert result_id == doc_id
                        assert mock_session.add.call_count == 3  # 1 document + 2 chunks
                        assert mock_session.commit.called
                        assert mock_insert.call_count == 2  # 2 embeddings
    
    @pytest.mark.asyncio
    async def test_ingest_document_with_specific_model(self, ingestion_service, sample_chunks, mock_session):
        """Test document ingestion with specific embedding model"""
        with patch.object(ingestion_service.chunker, 'split_text', return_value=sample_chunks):
            with patch('src.rag_system.services.ingestion.embedding_service') as mock_emb:
                mock_emb.embed_texts.return_value = ([[0.1] * 384, [0.2] * 384], 384)
                
                with patch('src.rag_system.services.ingestion.db') as mock_db:
                    mock_db.get_session.return_value.__aenter__.return_value = mock_session
                    mock_db.insert_embedding.return_value = uuid4()
                    
                    mock_session.add.side_effect = lambda obj: setattr(obj, 'id', uuid4())
                    
                    await ingestion_service.ingest_document(
                        title="Test",
                        content="Test content",
                        embedding_model="all-MiniLM-L6-v2"
                    )
                    
                    # Verify correct model was used
                    mock_emb.embed_texts.assert_called_once()
                    call_args = mock_emb.embed_texts.call_args
                    assert call_args[1]['model_name'] == "all-MiniLM-L6-v2"
    
    @pytest.mark.asyncio
    async def test_ingest_document_embedding_error(self, ingestion_service, sample_chunks, mock_session):
        """Test document ingestion with embedding generation error"""
        with patch.object(ingestion_service.chunker, 'split_text', return_value=sample_chunks):
            with patch('src.rag_system.services.ingestion.embedding_service') as mock_emb:
                mock_emb.embed_texts.side_effect = Exception("Embedding error")
                
                with patch('src.rag_system.services.ingestion.db.get_session') as mock_db:
                    mock_db.return_value.__aenter__.return_value = mock_session
                    
                    with pytest.raises(IngestionError) as exc_info:
                        await ingestion_service.ingest_document(
                            title="Test",
                            content="Test content"
                        )
                    
                    assert "Failed to ingest document" in str(exc_info.value)
                    assert mock_session.rollback.called
    
    @pytest.mark.asyncio
    async def test_ingest_documents_batch(self, ingestion_service):
        """Test batch document ingestion"""
        documents = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
            {"title": "Doc 3", "content": "Content 3"}
        ]
        
        with patch.object(ingestion_service, 'ingest_document') as mock_ingest:
            mock_ingest.side_effect = [uuid4(), uuid4(), uuid4()]
            
            result_ids = await ingestion_service.ingest_documents(documents)
            
            assert len(result_ids) == 3
            assert mock_ingest.call_count == 3
    
    @pytest.mark.asyncio
    async def test_update_document_embeddings_success(self, ingestion_service, mock_session):
        """Test updating document embeddings with new model"""
        doc_id = uuid4()
        chunks = [
            Mock(id=uuid4(), content="Chunk 1", chunk_index=0),
            Mock(id=uuid4(), content="Chunk 2", chunk_index=1)
        ]
        
        with patch('src.rag_system.services.ingestion.embedding_service') as mock_emb:
            mock_emb.get_model_dimension.return_value = 768
            mock_emb.embed_texts.return_value = ([[0.1] * 768, [0.2] * 768], 768)
            
            with patch('src.rag_system.services.ingestion.db') as mock_db:
                mock_db.get_session.return_value.__aenter__.return_value = mock_session
                mock_db.insert_embedding.return_value = uuid4()
                
                # Mock chunk retrieval
                mock_result = Mock()
                mock_result.scalars.return_value.all.return_value = chunks
                mock_session.execute.return_value = mock_result
                
                await ingestion_service.update_document_embeddings(
                    document_id=doc_id,
                    model_name="all-mpnet-base-v2"
                )
                
                # Verify embeddings were updated
                assert mock_emb.embed_texts.called
                assert mock_db.insert_embedding.call_count == 2
    
    @pytest.mark.asyncio
    async def test_update_document_embeddings_no_chunks(self, ingestion_service, mock_session):
        """Test updating embeddings for document with no chunks"""
        doc_id = uuid4()
        
        with patch('src.rag_system.services.ingestion.db.get_session') as mock_db:
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock empty chunk retrieval
            mock_result = Mock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute.return_value = mock_result
            
            with pytest.raises(IngestionError) as exc_info:
                await ingestion_service.update_document_embeddings(doc_id)
            
            assert f"No chunks found for document {doc_id}" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_document_embedding_status(self, ingestion_service, mock_session):
        """Test getting document embedding status"""
        doc_id = uuid4()
        doc = Mock(
            id=doc_id,
            title="Test Document",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        chunks = [
            Mock(id=uuid4()),
            Mock(id=uuid4())
        ]
        
        embeddings_data = [
            (Mock(embedding_version=1), Mock(model_name="text-embedding-ada-002", dimension=1536)),
            (Mock(embedding_version=1), Mock(model_name="text-embedding-ada-002", dimension=1536)),
            (Mock(embedding_version=1), Mock(model_name="all-MiniLM-L6-v2", dimension=384))
        ]
        
        with patch('src.rag_system.services.ingestion.db.get_session') as mock_db:
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock document query
            doc_result = Mock()
            doc_result.scalar_one_or_none.return_value = doc
            
            # Mock chunk query
            chunk_result = Mock()
            chunk_result.scalars.return_value.all.return_value = chunks
            
            # Mock embedding queries
            emb_results = [Mock(), Mock()]
            emb_results[0].__iter__ = Mock(return_value=iter(embeddings_data[:2]))
            emb_results[1].__iter__ = Mock(return_value=iter([embeddings_data[2]]))
            
            mock_session.execute.side_effect = [doc_result, chunk_result, emb_results[0], emb_results[1]]
            
            status = await ingestion_service.get_document_embedding_status(doc_id)
            
            assert status["document_id"] == str(doc_id)
            assert status["title"] == "Test Document"
            assert status["total_chunks"] == 2
            assert "text-embedding-ada-002" in status["embeddings"]
            assert status["embeddings"]["text-embedding-ada-002"]["count"] == 2
            assert status["embeddings"]["text-embedding-ada-002"]["dimension"] == 1536
    
    @pytest.mark.asyncio
    async def test_get_document_embedding_status_not_found(self, ingestion_service, mock_session):
        """Test getting status for non-existent document"""
        doc_id = uuid4()
        
        with patch('src.rag_system.services.ingestion.db.get_session') as mock_db:
            mock_db.return_value.__aenter__.return_value = mock_session
            
            # Mock document not found
            doc_result = Mock()
            doc_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = doc_result
            
            with pytest.raises(ValueError) as exc_info:
                await ingestion_service.get_document_embedding_status(doc_id)
            
            assert f"Document {doc_id} not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_migrate_to_new_model(self, ingestion_service, mock_session):
        """Test migrating documents to new embedding model"""
        doc_ids = [uuid4(), uuid4()]
        documents = [Mock(id=doc_ids[0]), Mock(id=doc_ids[1])]
        
        with patch('src.rag_system.services.ingestion.embedding_service') as mock_emb:
            mock_emb.get_model_dimension.side_effect = [1536, 3072]  # source, target dimensions
            
            with patch('src.rag_system.services.ingestion.db.get_session') as mock_db:
                mock_db.return_value.__aenter__.return_value = mock_session
                
                # Mock document query
                doc_result = Mock()
                doc_result.scalars.return_value.all.return_value = documents
                mock_session.execute.return_value = doc_result
                
                with patch.object(ingestion_service, 'update_document_embeddings') as mock_update:
                    mock_update.return_value = None
                    
                    await ingestion_service.migrate_to_new_model(
                        source_model="text-embedding-ada-002",
                        target_model="text-embedding-3-large",
                        document_ids=doc_ids
                    )
                    
                    # Verify each document was updated
                    assert mock_update.call_count == 2
                    for doc_id in doc_ids:
                        mock_update.assert_any_call(
                            doc_id,
                            model_name="text-embedding-3-large",
                            batch_size=100
                        )
    
    @pytest.mark.asyncio
    async def test_migrate_to_new_model_partial_failure(self, ingestion_service, mock_session):
        """Test migration with some documents failing"""
        documents = [Mock(id=uuid4()), Mock(id=uuid4())]
        
        with patch('src.rag_system.services.ingestion.embedding_service') as mock_emb:
            mock_emb.get_model_dimension.side_effect = [384, 768]
            
            with patch('src.rag_system.services.ingestion.db.get_session') as mock_db:
                mock_db.return_value.__aenter__.return_value = mock_session
                
                doc_result = Mock()
                doc_result.scalars.return_value.all.return_value = documents
                mock_session.execute.return_value = doc_result
                
                with patch.object(ingestion_service, 'update_document_embeddings') as mock_update:
                    # First succeeds, second fails
                    mock_update.side_effect = [None, Exception("Update failed")]
                    
                    await ingestion_service.migrate_to_new_model(
                        source_model="all-MiniLM-L6-v2",
                        target_model="all-mpnet-base-v2"
                    )
                    
                    # Should continue despite failure
                    assert mock_update.call_count == 2


class TestIngestionIntegration:
    """Integration tests for document ingestion"""
    
    @pytest.mark.asyncio
    async def test_full_ingestion_flow(self):
        """Test complete document ingestion flow"""
        service = DocumentIngestionService()
        
        # Mock all dependencies
        with patch('src.rag_system.utils.chunking.TextChunker.split_text') as mock_chunk:
            mock_chunk.return_value = [
                TextChunk(content="Test chunk", start_char=0, end_char=10, chunk_index=0)
            ]
            
            with patch('src.rag_system.services.embeddings.embedding_service') as mock_emb:
                mock_emb.current_model_name = "test-model"
                mock_emb.embed_texts.return_value = ([[0.1] * 1536], 1536)
                
                with patch('src.rag_system.db.pgvector.db') as mock_db:
                    # Mock session context
                    mock_session = AsyncMock()
                    mock_session.add = Mock()
                    mock_session.flush = AsyncMock()
                    mock_session.commit = AsyncMock()
                    
                    mock_db.get_session.return_value.__aenter__.return_value = mock_session
                    mock_db.insert_embedding.return_value = uuid4()
                    
                    # Mock ID assignment
                    doc_id = uuid4()
                    mock_session.add.side_effect = lambda obj: setattr(obj, 'id', doc_id)
                    
                    result = await service.ingest_document(
                        title="Integration Test",
                        content="Test chunk content"
                    )
                    
                    assert result == doc_id
                    assert mock_session.commit.called