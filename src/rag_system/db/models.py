# src/rag_system/db/models.py
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, JSON, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    source_url = Column(Text)
    document_type = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata = Column(JSON, default={})
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id', ondelete='CASCADE'))
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    start_char = Column(Integer)
    end_char = Column(Integer)
    token_count = Column(Integer)
    chunk_metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("ChunkEmbedding", back_populates="chunk", cascade="all, delete-orphan")

class EmbeddingModel(Base):
    """Track embedding models and their dimensions"""
    __tablename__ = 'embedding_models'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), unique=True, nullable=False)
    dimension = Column(Integer, nullable=False)
    is_active = Column(Integer, default=1)  # Boolean as Integer for SQLite compatibility
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint('dimension > 0', name='positive_dimension'),
    )

class ChunkEmbedding(Base):
    __tablename__ = 'chunk_embeddings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey('document_chunks.id', ondelete='CASCADE'))
    model_id = Column(UUID(as_uuid=True), ForeignKey('embedding_models.id'))
    embedding_version = Column(Integer, default=1)
    # Note: We'll create the vector column dynamically based on model dimension
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embeddings")
    model = relationship("EmbeddingModel")