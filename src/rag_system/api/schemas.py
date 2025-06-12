# src/rag_system/api/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    embedding_model: str
    llm_model: str


class IngestionRequest(BaseModel):
    """Request model for document ingestion"""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    source_url: Optional[str] = Field(None, description="Source URL of the document")
    document_type: Optional[str] = Field("text", description="Type of document")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "title": "Introduction to RAG",
            "content": "Retrieval-Augmented Generation (RAG) is a technique...",
            "source_url": "https://example.com/rag-intro",
            "document_type": "article",
            "metadata": {"author": "John Doe", "date": "2024-01-01"}
        }
    })


class IngestionResponse(BaseModel):
    """Response model for document ingestion"""
    document_id: Optional[UUID]
    status: str
    message: str


class SearchRequest(BaseModel):
    """Request model for document search"""
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(5, ge=1, le=100, description="Number of results to return")
    threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    rerank: Optional[bool] = Field(False, description="Whether to apply reranking")
    rerank_model: Optional[str] = Field(None, description="Reranking model to use")
    rerank_top_k: Optional[int] = Field(None, description="Number of results after reranking")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What is PgVector?",
            "top_k": 5,
            "threshold": 0.7,
            "filters": {"document_type": "article"},
            "rerank": True,
            "rerank_model": "ms-marco-MiniLM-L-6-v2"
        }
    })


class SearchResult(BaseModel):
    """Individual search result"""
    document_id: str
    document_title: str
    content: str
    chunk_index: int
    similarity_score: float
    rerank_score: Optional[float] = Field(None, description="Cross-encoder reranking score")
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    """Response model for document search"""
    query: str
    results: List[SearchResult]
    total_results: int

class RerankingModelInfo(BaseModel):
    """Information about a reranking model"""
    model_name: str
    batch_size: int
    max_length: int
    normalize_scores: bool
    is_default: bool
    is_loaded: bool

class RerankingStatusResponse(BaseModel):
    """Response for reranking status"""
    enabled: bool
    default_model: str
    available_models: Dict[str, RerankingModelInfo]


class UpdateRerankingRequest(BaseModel):
    """Request to update reranking settings"""
    enabled: bool
    model: Optional[str] = Field(None, description="Reranking model to use as default")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "enabled": True,
            "model": "ms-marco-MiniLM-L-12-v2"
        }
    })

class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="User query")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of documents to retrieve")
    model: Optional[str] = Field(None, description="LLM model to use for generation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum tokens to generate")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for retrieval")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "How does PgVector indexing work?",
            "top_k": 5,
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 500
        }
    })


class Source(BaseModel):
    """Source document information"""
    document_id: str
    document_title: str
    chunk_content: str
    similarity_score: float


class QueryResponse(BaseModel):
    """Response model for RAG query"""
    answer: str
    sources: List[Source]
    metadata: Dict[str, Any]
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "answer": "PgVector uses HNSW and IVFFlat indexes for efficient similarity search...",
            "sources": [
                {
                    "document_id": "123e4567-e89b-12d3-a456-426614174000",
                    "document_title": "PgVector Basics",
                    "chunk_content": "PgVector supports multiple index types...",
                    "similarity_score": 0.92
                }
            ],
            "metadata": {
                "model_used": "gpt-4o-mini",
                "tokens_used": 256,
                "retrieval_count": 5
            }
        }
    })


class DocumentRequest(BaseModel):
    """Request model for document operations"""
    document_id: UUID
    operation: str = Field(..., description="Operation to perform", pattern="^(delete|update)$")


class DocumentResponse(BaseModel):
    """Response model for document operations"""
    document_id: UUID
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)