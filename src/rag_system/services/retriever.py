# src/rag_system/services/retriever.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import CrossEncoder
import asyncio
from functools import lru_cache

from ..db.pgvector import db
from ..services.embeddings import embedding_service
from ..core.config import settings
from ..utils.monitoring import logger, query_duration, query_counter, embedding_generation_duration
from ..utils.exceptions import RetrievalError, ValidationError
from ..utils.circuit_breaker import CircuitBreakers, create_circuit_breaker


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
    rerank_score: Optional[float] = None  # Added for reranking


@dataclass
class RerankingConfig:
    """Configuration for reranking models"""
    model_name: str
    batch_size: int = 32
    max_length: int = 512
    normalize_scores: bool = True
    cache_model: bool = True


class RerankingModel:
    """Wrapper for cross-encoder reranking models"""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self._model = None
        self._circuit_breaker = create_circuit_breaker(
            name=f"reranker_{config.model_name}",
            failure_threshold=3,
            recovery_timeout=30,
            timeout=10.0
        )
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model"""
        if self._model is None:
            logger.info(f"Loading reranking model: {self.config.model_name}")
            self._model = CrossEncoder(self.config.model_name, max_length=self.config.max_length)
        return self._model
    
    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """Rerank chunks using cross-encoder"""
        if not chunks:
            return chunks
        
        try:
            # Prepare pairs for the cross-encoder
            pairs = [(query, chunk.content) for chunk in chunks]
            
            # Score in batches
            all_scores = []
            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i:i + self.config.batch_size]
                
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                batch_scores = await self._circuit_breaker.call_async(
                    loop.run_in_executor,
                    None,
                    self.model.predict,
                    batch
                )
                all_scores.extend(batch_scores)
            
            # Normalize scores if requested
            if self.config.normalize_scores:
                scores_array = np.array(all_scores)
                # Min-max normalization to [0, 1]
                if scores_array.max() > scores_array.min():
                    all_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
                else:
                    all_scores = scores_array
            
            # Update chunks with rerank scores
            for chunk, score in zip(chunks, all_scores):
                chunk.rerank_score = float(score)
            
            # Sort by rerank score
            reranked_chunks = sorted(chunks, key=lambda x: x.rerank_score, reverse=True)
            
            # Return top_k if specified
            if top_k:
                reranked_chunks = reranked_chunks[:top_k]
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order on failure
            return chunks[:top_k] if top_k else chunks


class RetrieverService:
    """Service for retrieving relevant documents using vector similarity search"""
    
    # Available reranking models
    RERANKING_MODELS = {
        "ms-marco-MiniLM-L-6-v2": RerankingConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            batch_size=32,
            normalize_scores=True
        ),
        "ms-marco-MiniLM-L-12-v2": RerankingConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            batch_size=16,
            normalize_scores=True
        ),
        "ms-marco-TinyBERT-L-2-v2": RerankingConfig(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
            batch_size=64,
            normalize_scores=True
        ),
        "qnli-electra-base": RerankingConfig(
            model_name="cross-encoder/qnli-electra-base",
            batch_size=16,
            normalize_scores=True
        )
    }
    
    def __init__(self):
        self.reranking_enabled = False
        self._model_dimension_cache = {}
        self._reranking_models: Dict[str, RerankingModel] = {}
        self._default_reranking_model = "ms-marco-MiniLM-L-6-v2"
        
    async def search(
        self,
        query: str,
        top_k: int = None,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        rerank: bool = None,
        rerank_model: Optional[str] = None,
        rerank_top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Search for relevant documents based on query
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Metadata filters for documents
            model_name: Specific embedding model to use
            rerank: Whether to apply reranking (overrides self.reranking_enabled)
            rerank_model: Specific reranking model to use
            rerank_top_k: Number of results to return after reranking
            
        Returns:
            List of retrieved chunks sorted by relevance
        """
        # Track metrics
        query_counter.inc()
        
        # Use default top_k if not specified
        if top_k is None:
            top_k = settings.retrieval_top_k
        
        # Determine if we should rerank
        should_rerank = rerank if rerank is not None else self.reranking_enabled
        
        # If reranking, retrieve more candidates
        retrieval_top_k = top_k * 3 if should_rerank else top_k
        
        # Use specified model or current default
        search_model = model_name or embedding_service.current_model_name
        
        try:
            # Log search request
            logger.info(
                "retrieval_search",
                query=query[:100],
                top_k=top_k,
                threshold=threshold,
                filters=filters,
                model=search_model,
                rerank=should_rerank,
                rerank_model=rerank_model if should_rerank else None
            )
            
            # Generate query embedding with dimension info
            query_embedding, dimension = await embedding_service.embed_text(
                query, 
                model_name=search_model
            )
            
            # Validate dimension matches expected
            expected_dim = embedding_service.get_model_dimension(search_model)
            if dimension != expected_dim:
                raise ValidationError(
                    f"Embedding dimension mismatch: got {dimension}, "
                    f"expected {expected_dim} for model {search_model}"
                )
            
            # Perform similarity search with model name
            with query_duration.time():
                results = await db.similarity_search(
                    embedding=query_embedding,
                    model_name=search_model,
                    top_k=retrieval_top_k,
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
            if should_rerank and len(retrieved_chunks) > 0:
                retrieved_chunks = await self._rerank_results(
                    query=query,
                    chunks=retrieved_chunks,
                    rerank_top_k=rerank_top_k or top_k,
                    model_name=rerank_model
                )
            else:
                # Just take top_k without reranking
                retrieved_chunks = retrieved_chunks[:top_k]
            
            logger.info(
                "retrieval_complete",
                query=query[:100],
                results_count=len(retrieved_chunks),
                top_score=retrieved_chunks[0].similarity_score if retrieved_chunks else 0,
                top_rerank_score=retrieved_chunks[0].rerank_score if retrieved_chunks and retrieved_chunks[0].rerank_score else None,
                model=search_model,
                dimension=dimension,
                reranked=should_rerank
            )
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(
                "retrieval_error",
                query=query[:100],
                error=str(e),
                model=search_model
            )
            raise RetrievalError(f"Search failed: {str(e)}")
    
    async def search_multi_model(
        self,
        query: str,
        model_names: List[str],
        top_k: int = None,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        aggregation: str = "union",  # "union" or "intersection"
        rerank: bool = True,
        rerank_model: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Search using multiple embedding models and aggregate results
        
        Args:
            query: Search query text
            model_names: List of embedding models to use
            top_k: Number of results to return per model
            threshold: Minimum similarity threshold
            filters: Metadata filters
            aggregation: How to combine results ("union" or "intersection")
            rerank: Whether to apply reranking to final results
            rerank_model: Specific reranking model to use
            
        Returns:
            Aggregated list of retrieved chunks
        """
        if not model_names:
            raise ValueError("At least one model name must be provided")
        
        # Use default top_k if not specified
        if top_k is None:
            top_k = settings.retrieval_top_k
        
        # Collect results from each model
        all_results = {}
        chunk_scores = {}  # chunk_id -> {model -> score}
        
        for model_name in model_names:
            try:
                results = await self.search(
                    query=query,
                    top_k=top_k * 2,  # Get more for aggregation
                    threshold=threshold,
                    filters=filters,
                    model_name=model_name,
                    rerank=False  # Don't rerank individual results
                )
                
                for chunk in results:
                    chunk_id = chunk.chunk_id
                    
                    if chunk_id not in all_results:
                        all_results[chunk_id] = chunk
                        chunk_scores[chunk_id] = {}
                    
                    chunk_scores[chunk_id][model_name] = chunk.similarity_score
                    
            except Exception as e:
                logger.error(f"Search failed for model {model_name}: {e}")
                if len(model_names) == 1:
                    raise
        
        # Aggregate results based on strategy
        final_results = []
        
        if aggregation == "intersection":
            # Only keep chunks found by all models
            for chunk_id, chunk in all_results.items():
                if len(chunk_scores[chunk_id]) == len(model_names):
                    # Average the scores
                    avg_score = sum(chunk_scores[chunk_id].values()) / len(model_names)
                    chunk.similarity_score = avg_score
                    final_results.append(chunk)
        else:  # union
            # Keep all chunks, use average score where available
            for chunk_id, chunk in all_results.items():
                scores = chunk_scores[chunk_id].values()
                chunk.similarity_score = sum(scores) / len(scores)
                final_results.append(chunk)
        
        # Sort by score
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Apply reranking if requested
        if rerank and final_results:
            final_results = await self._rerank_results(
                query=query,
                chunks=final_results,
                rerank_top_k=top_k,
                model_name=rerank_model
            )
        else:
            final_results = final_results[:top_k]
        
        return final_results
    
    async def search_hybrid(
        self,
        query: str,
        top_k: int = None,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        rerank: bool = True,
        rerank_model: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Hybrid search combining vector similarity and keyword matching
        
        Args:
            query: Search query text
            top_k: Number of results to return
            keyword_weight: Weight for keyword matching (0-1)
            filters: Metadata filters
            model_name: Embedding model to use
            rerank: Whether to apply reranking
            rerank_model: Specific reranking model to use
            
        Returns:
            List of retrieved chunks with hybrid scoring
        """
        # Get vector search results
        vector_results = await self.search(
            query=query,
            top_k=top_k * 2 if top_k else 20,  # Get more for merging
            filters=filters,
            model_name=model_name,
            rerank=rerank,
            rerank_model=rerank_model
        )
        
        # TODO: Implement keyword search using PostgreSQL full-text search
        # For now, return vector results
        return vector_results[:top_k] if top_k else vector_results
    
    async def get_available_models_for_search(self) -> List[Dict[str, Any]]:
        """Get list of models that have embeddings in the database"""
        async with db.get_session() as session:
            from sqlalchemy import select, func
            from ..db.models import EmbeddingModel, ChunkEmbedding
            
            # Query to get models with embedding counts
            result = await session.execute(
                select(
                    EmbeddingModel.model_name,
                    EmbeddingModel.dimension,
                    func.count(ChunkEmbedding.id).label('embedding_count')
                )
                .join(ChunkEmbedding, EmbeddingModel.id == ChunkEmbedding.model_id)
                .group_by(EmbeddingModel.model_name, EmbeddingModel.dimension)
                .having(func.count(ChunkEmbedding.id) > 0)
            )
            
            models = []
            for row in result:
                models.append({
                    "model_name": row.model_name,
                    "dimension": row.dimension,
                    "embedding_count": row.embedding_count,
                    "is_current": row.model_name == embedding_service.current_model_name
                })
            
            return models
    
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
        rerank_top_k: int = None,
        model_name: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Rerank results using a cross-encoder model
        
        Args:
            query: Original query
            chunks: Initial retrieved chunks
            rerank_top_k: Number of results to return after reranking
            model_name: Specific reranking model to use
            
        Returns:
            Reranked chunks
        """
        model_name = model_name or self._default_reranking_model
        
        # Validate model name
        if model_name not in self.RERANKING_MODELS:
            logger.warning(
                f"Unknown reranking model: {model_name}. "
                f"Using default: {self._default_reranking_model}"
            )
            model_name = self._default_reranking_model
        
        # Get or create reranking model
        if model_name not in self._reranking_models:
            config = self.RERANKING_MODELS[model_name]
            self._reranking_models[model_name] = RerankingModel(config)
        
        reranker = self._reranking_models[model_name]
        
        # Time the reranking
        with embedding_generation_duration.labels(model=f"rerank_{model_name}").time():
            reranked_chunks = await reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=rerank_top_k
            )
        
        logger.info(
            "reranking_complete",
            query=query[:100],
            input_chunks=len(chunks),
            output_chunks=len(reranked_chunks),
            model=model_name,
            top_score_before=chunks[0].similarity_score if chunks else 0,
            top_score_after=reranked_chunks[0].rerank_score if reranked_chunks else 0
        )
        
        return reranked_chunks
    
    def enable_reranking(self, enabled: bool = True, model: Optional[str] = None):
        """Enable or disable reranking with optional model selection"""
        self.reranking_enabled = enabled
        if model and model in self.RERANKING_MODELS:
            self._default_reranking_model = model
        logger.info(
            "reranking_status_changed",
            enabled=enabled,
            model=self._default_reranking_model if enabled else None
        )
    
    def get_reranking_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available reranking models and their configurations"""
        return {
            name: {
                "model_name": config.model_name,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "normalize_scores": config.normalize_scores,
                "is_default": name == self._default_reranking_model,
                "is_loaded": name in self._reranking_models
            }
            for name, config in self.RERANKING_MODELS.items()
        }


# Global instance
retriever = RetrieverService()