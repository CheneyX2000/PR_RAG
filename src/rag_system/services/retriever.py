# src/rag_system/services/retriever.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from ..db.pgvector import db
from ..services.embeddings import embedding_service
from ..core.config import settings
from ..utils.monitoring import logger, query_duration, query_counter


@dataclass
class RetrievedChunk:
    """Retrieved document chunk with metadata"""
    document_id: str
    document_title: str
    chunk_id: str
    content: str
    chunk_index: int
    similarity_score: float
    metadata: Dict[str, Any] = None


class RetrieverService:
    """Service for retrieving relevant documents using vector similarity search"""
    
    def __init__(self):
        self.reranking_enabled = False  # Can be enhanced later
        
    async def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Search for relevant documents based on query
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Metadata filters for documents
            model_name: Specific embedding model to use
            
        Returns:
            List of retrieved chunks sorted by relevance
        """
        # Track metrics
        query_counter.inc()
        
        # Use default top_k if not specified
        if top_k is None:
            top_k = settings.retrieval_top_k
        
        try:
            # Log search request
            logger.info(
                "retrieval_search",
                query=query[:100],  # Log truncated query
                top_k=top_k,
                threshold=threshold,
                filters=filters
            )
            
            # Generate query embedding
            if model_name and model_name != embedding_service.current_model_name:
                # Temporarily switch model for this query
                original_model = embedding_service.current_model_name
                embedding_service.switch_model(model_name)
                query_embedding = await embedding_service.embed_text(query)
                embedding_service.switch_model(original_model)
            else:
                query_embedding = await embedding_service.embed_text(query)
            
            # Perform similarity search
            with query_duration.time():
                results = await db.similarity_search(
                    embedding=query_embedding,
                    top_k=top_k,
                    threshold=threshold,
                    filters=filters
                )
            
            # Convert results to RetrievedChunk objects
            retrieved_chunks = []
            for result in results:
                chunk = RetrievedChunk(
                    document_id=result['document_id'],
                    document_title=result['title'],
                    chunk_id=result['chunk_id'],
                    content=result['content'],
                    chunk_index=result['chunk_index'],
                    similarity_score=result['similarity'],
                    metadata=result.get('metadata', {})
                )
                retrieved_chunks.append(chunk)
            
            # Apply reranking if enabled
            if self.reranking_enabled and len(retrieved_chunks) > 0:
                retrieved_chunks = await self._rerank_results(query, retrieved_chunks)
            
            logger.info(
                "retrieval_complete",
                query=query[:100],
                results_count=len(retrieved_chunks),
                top_score=retrieved_chunks[0].similarity_score if retrieved_chunks else 0
            )
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(
                "retrieval_error",
                query=query[:100],
                error=str(e)
            )
            raise
    
    async def search_hybrid(
        self,
        query: str,
        top_k: int = None,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Hybrid search combining vector similarity and keyword matching
        
        Args:
            query: Search query text
            top_k: Number of results to return
            keyword_weight: Weight for keyword matching (0-1)
            filters: Metadata filters
            
        Returns:
            List of retrieved chunks with hybrid scoring
        """
        # Get vector search results
        vector_results = await self.search(
            query=query,
            top_k=top_k * 2 if top_k else 20,  # Get more for merging
            filters=filters
        )
        
        # TODO: Implement keyword search and merge results
        # For now, return vector results
        return vector_results[:top_k] if top_k else vector_results
    
    async def get_context_window(
        self,
        chunk_ids: List[str],
        window_size: int = 1
    ) -> Dict[str, List[RetrievedChunk]]:
        """
        Get surrounding chunks for context expansion
        
        Args:
            chunk_ids: List of chunk IDs
            window_size: Number of chunks before/after to include
            
        Returns:
            Dictionary mapping chunk_id to list of context chunks
        """
        # TODO: Implement context window expansion
        # This would fetch chunks before and after the retrieved chunks
        # from the same document to provide more context
        return {}
    
    async def _rerank_results(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        rerank_top_k: int = None
    ) -> List[RetrievedChunk]:
        """
        Rerank results using a more sophisticated model
        
        Args:
            query: Original query
            chunks: Initial retrieved chunks
            rerank_top_k: Number of results to return after reranking
            
        Returns:
            Reranked chunks
        """
        # TODO: Implement reranking using cross-encoder or LLM
        # For now, return original results
        return chunks[:rerank_top_k] if rerank_top_k else chunks
    
    def enable_reranking(self, enabled: bool = True):
        """Enable or disable reranking"""
        self.reranking_enabled = enabled
        logger.info("reranking_status_changed", enabled=enabled)


# Global instance
retriever = RetrieverService()