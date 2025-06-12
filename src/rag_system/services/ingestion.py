# src/rag_system/services/ingestion.py
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..db.models import Document, DocumentChunk, ChunkEmbedding, EmbeddingModel
from ..db.pgvector import db
from ..utils.chunking import TextChunker, TextChunk
from .embeddings import embedding_service
from ..core.config import settings
from ..utils.monitoring import logger, document_ingestion_counter, document_ingestion_duration
from ..utils.exceptions import IngestionError

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
        metadata: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[str] = None
    ) -> UUID:
        """Ingest a single document with dynamic embedding support"""
        
        # Use specified model or current default
        model_name = embedding_model or embedding_service.current_model_name
        
        try:
            with document_ingestion_duration.time():
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
                            token_count=len(chunk.content.split()),
                            chunk_metadata=chunk.metadata or {}
                        )
                        session.add(chunk_record)
                        chunk_records.append(chunk_record)
                        chunk_texts.append(chunk.content)
                    
                    await session.flush()  # Get chunk IDs
                    
                    # Generate embeddings with dimension info
                    embeddings, dimension = await embedding_service.embed_texts(
                        chunk_texts, 
                        model_name=model_name
                    )
                    
                    # Log embedding generation
                    logger.info(
                        f"Generated {len(embeddings)} embeddings "
                        f"with model {model_name} (dimension={dimension})"
                    )
                    
                    # Store embeddings using the new system
                    for chunk_record, embedding in zip(chunk_records, embeddings):
                        await db.insert_embedding(
                            chunk_id=chunk_record.id,
                            model_name=model_name,
                            embedding=embedding,
                            dimension=dimension,
                            version=1
                        )
                    
                    await session.commit()
                    
                    # Update metrics
                    document_ingestion_counter.labels(document_type=document_type).inc()
                    
                    logger.info(
                        f"Successfully ingested document '{title}' "
                        f"with {len(chunks)} chunks using {model_name}"
                    )
                    
                    return document.id
                    
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise IngestionError(f"Failed to ingest document: {str(e)}")
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]],
        embedding_model: Optional[str] = None
    ) -> List[UUID]:
        """Ingest multiple documents"""
        document_ids = []
        
        for doc in documents:
            doc_id = await self.ingest_document(
                embedding_model=embedding_model,
                **doc
            )
            document_ids.append(doc_id)
        
        return document_ids
    
    async def update_document_embeddings(
        self,
        document_id: UUID,
        model_name: Optional[str] = None,
        batch_size: int = 50
    ):
        """Update embeddings for a document with a new model"""
        if model_name:
            # Validate model exists
            dimension = embedding_service.get_model_dimension(model_name)
            logger.info(
                f"Updating embeddings for document {document_id} "
                f"to model {model_name} (dimension={dimension})"
            )
        else:
            model_name = embedding_service.current_model_name
            dimension = embedding_service.get_model_dimension(model_name)
        
        try:
            async with db.get_session() as session:
                # Get all chunks for the document
                result = await session.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document_id)
                    .order_by(DocumentChunk.chunk_index)
                )
                chunks = result.scalars().all()
                
                if not chunks:
                    raise IngestionError(f"No chunks found for document {document_id}")
                
                # Process in batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    chunk_texts = [chunk.content for chunk in batch]
                    
                    # Generate new embeddings
                    embeddings, dimension = await embedding_service.embed_texts(
                        chunk_texts,
                        model_name=model_name
                    )
                    
                    # Update embeddings
                    for chunk, embedding in zip(batch, embeddings):
                        await db.insert_embedding(
                            chunk_id=chunk.id,
                            model_name=model_name,
                            embedding=embedding,
                            dimension=dimension,
                            version=1  # Could increment if updating existing
                        )
                
                logger.info(
                    f"Successfully updated {len(chunks)} embeddings "
                    f"for document {document_id} with model {model_name}"
                )
                
        except Exception as e:
            logger.error(f"Failed to update embeddings: {e}")
            raise IngestionError(f"Failed to update embeddings: {str(e)}")
    
    async def get_document_embedding_status(
        self, 
        document_id: UUID
    ) -> Dict[str, Any]:
        """Get embedding status for a document"""
        async with db.get_session() as session:
            # Get document info
            doc_result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = doc_result.scalar_one_or_none()
            
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Get chunk count
            chunk_result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
            )
            chunks = chunk_result.scalars().all()
            
            # Get embedding info
            embedding_info = {}
            
            for chunk in chunks:
                emb_result = await session.execute(
                    select(ChunkEmbedding, EmbeddingModel)
                    .join(EmbeddingModel)
                    .where(ChunkEmbedding.chunk_id == chunk.id)
                )
                
                for emb, model in emb_result:
                    if model.model_name not in embedding_info:
                        embedding_info[model.model_name] = {
                            "dimension": model.dimension,
                            "count": 0,
                            "versions": set()
                        }
                    
                    embedding_info[model.model_name]["count"] += 1
                    embedding_info[model.model_name]["versions"].add(emb.embedding_version)
            
            # Convert sets to lists for JSON serialization
            for model_info in embedding_info.values():
                model_info["versions"] = sorted(list(model_info["versions"]))
            
            return {
                "document_id": str(document_id),
                "title": document.title,
                "total_chunks": len(chunks),
                "embeddings": embedding_info,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat()
            }
    
    async def migrate_to_new_model(
        self,
        source_model: str,
        target_model: str,
        document_ids: Optional[List[UUID]] = None,
        batch_size: int = 100
    ):
        """Migrate documents from one embedding model to another"""
        source_dim = embedding_service.get_model_dimension(source_model)
        target_dim = embedding_service.get_model_dimension(target_model)
        
        logger.info(
            f"Starting migration from {source_model} (dim={source_dim}) "
            f"to {target_model} (dim={target_dim})"
        )
        
        # This would be a background task in production
        # For now, it's a placeholder showing the structure
        
        async with db.get_session() as session:
            # Get documents to migrate
            query = select(Document)
            if document_ids:
                query = query.where(Document.id.in_(document_ids))
            
            result = await session.execute(query)
            documents = result.scalars().all()
            
            migrated_count = 0
            
            for document in documents:
                try:
                    await self.update_document_embeddings(
                        document.id,
                        model_name=target_model,
                        batch_size=batch_size
                    )
                    migrated_count += 1
                    
                    if migrated_count % 10 == 0:
                        logger.info(f"Migrated {migrated_count}/{len(documents)} documents")
                        
                except Exception as e:
                    logger.error(f"Failed to migrate document {document.id}: {e}")
            
            logger.info(
                f"Migration complete: {migrated_count}/{len(documents)} documents "
                f"migrated to {target_model}"
            )

# Global instance
ingestion_service = DocumentIngestionService()