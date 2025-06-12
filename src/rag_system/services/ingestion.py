# src/rag_system/services/ingestion.py
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..db.models import Document, DocumentChunk, ChunkEmbedding
from ..db.pgvector import db
from ..utils.chunking import TextChunker, TextChunk
from .embeddings import embedding_service
from ..core.config import settings

class DocumentIngestionService:
    """Service for ingesting documents into the RAG system"""
    
    def __init__(self):
        self.chunker = TextChunker(
            chunk_size=settings.max_chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    async def ingest_document(
        self,
        title: str,
        content: str,
        source_url: Optional[str] = None,
        document_type: Optional[str] = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Ingest a single document"""
        async with db.get_session() as session:
            # Create document
            document = Document(
                title=title,
                content=content,
                source_url=source_url,
                document_type=document_type,
                metadata=metadata or {}
            )
            session.add(document)
            await session.flush()  # Get the document ID
            
            # Split into chunks
            chunks = self.chunker.split_text(content)
            
            # Create chunk records
            chunk_texts = []
            chunk_records = []
            
            for chunk in chunks:
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    token_count=len(chunk.content.split()),  # Simple token count
                    chunk_metadata=chunk.metadata or {}
                )
                session.add(chunk_record)
                chunk_records.append(chunk_record)
                chunk_texts.append(chunk.content)
            
            await session.flush()  # Get chunk IDs
            
            # Generate embeddings
            embeddings = await embedding_service.embed_texts(chunk_texts)
            
            # Create embedding records
            for chunk_record, embedding in zip(chunk_records, embeddings):
                embedding_record = ChunkEmbedding(
                    chunk_id=chunk_record.id,
                    model_name=embedding_service.current_model_name,
                    embedding_version=1,
                    embedding=embedding
                )
                session.add(embedding_record)
            
            await session.commit()
            
            print(f"Ingested document '{title}' with {len(chunks)} chunks")
            return document.id
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[UUID]:
        """Ingest multiple documents"""
        document_ids = []
        
        for doc in documents:
            doc_id = await self.ingest_document(**doc)
            document_ids.append(doc_id)
        
        return document_ids
    
    async def update_document_embeddings(
        self,
        document_id: UUID,
        model_name: Optional[str] = None
    ):
        """Update embeddings for a document with a new model"""
        if model_name:
            embedding_service.switch_model(model_name)
        
        async with db.get_session() as session:
            # Get all chunks for the document
            result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            )
            chunks = result.scalars().all()
            
            # Extract chunk texts
            chunk_texts = [chunk.content for chunk in chunks]
            
            # Generate new embeddings
            embeddings = await embedding_service.embed_texts(chunk_texts)
            
            # Update or create embedding records
            for chunk, embedding in zip(chunks, embeddings):
                # Check if embedding exists for this model
                result = await session.execute(
                    select(ChunkEmbedding)
                    .where(
                        ChunkEmbedding.chunk_id == chunk.id,
                        ChunkEmbedding.model_name == embedding_service.current_model_name
                    )
                )
                existing_embedding = result.scalar_one_or_none()
                
                if existing_embedding:
                    existing_embedding.embedding = embedding
                    existing_embedding.embedding_version += 1
                else:
                    embedding_record = ChunkEmbedding(
                        chunk_id=chunk.id,
                        model_name=embedding_service.current_model_name,
                        embedding_version=1,
                        embedding=embedding
                    )
                    session.add(embedding_record)
            
            await session.commit()

# Global instance
ingestion_service = DocumentIngestionService()