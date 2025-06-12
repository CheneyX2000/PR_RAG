# src/rag_system/api/routes.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
from uuid import UUID
from datetime import datetime
from ..utils.circuit_breaker import CircuitBreaker

from .schemas import (
    QueryRequest,
    QueryResponse,
    DocumentRequest,
    DocumentResponse,
    IngestionRequest,
    IngestionResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse,
    RerankingModelInfo,
    RerankingStatusResponse,
    UpdateRerankingRequest,
)
from ..services.retriever import retriever
from ..services.ingestion import ingestion_service
from ..services.generator import generator_service
from ..services.embeddings import embedding_service
from ..core.config import settings
from ..core.dependencies import (
    check_rate_limit,
    get_request_context,
    check_database_health,
    check_redis_health,
    RequestContext
)
from ..utils.monitoring import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db_healthy: bool = Depends(check_database_health),
    redis_healthy: bool = Depends(check_redis_health)
):
    """Check if the service is healthy"""
    # Check circuit breakers
    circuit_breaker_stats = CircuitBreaker.get_all_stats()
    open_breakers = [
        name for name, stats in circuit_breaker_stats.items() 
        if stats["state"] == "open"
    ]
    
    # Determine overall status
    if not db_healthy or open_breakers:
        status = "unhealthy"
    elif not redis_healthy:
        status = "degraded" 
    else:
        status = "healthy"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        embedding_model=settings.default_embedding_model,
        llm_model=settings.default_llm_model
    )

@router.delete("/documents/{document_id}")
async def delete_document(document_id: UUID):
    """Delete a document and all its chunks"""
    # TODO: Implement document deletion
    raise HTTPException(status_code=501, detail="Not implemented yet")

@router.get("/models/search", response_model=List[Dict[str, Any]])
async def get_search_models():
    """Get available embedding models for search"""
    try:
        models = await retriever.get_available_models_for_search()
        return models
    except Exception as e:
        logger.error(f"Failed to get search models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/available", response_model=Dict[str, int])
async def get_available_models():
    """Get all available embedding models and their dimensions"""
    try:
        models = embedding_service.get_available_models()
        return models
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{document_id}/embeddings")
async def get_document_embeddings(document_id: UUID):
    """Get embedding status for a document"""
    try:
        status = await ingestion_service.get_document_embedding_status(document_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get embedding status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestionResponse, dependencies=[Depends(check_rate_limit)])
async def ingest_document(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    context: RequestContext = Depends(get_request_context)
):
    """Ingest a new document into the RAG system"""
    try:
        logger.info(
            "document_ingestion_start",
            request_id=context.request_id,
            title=request.title
        )
        
        # Ingest document
        document_id = await ingestion_service.ingest_document(
            title=request.title,
            content=request.content,
            source_url=request.source_url,
            document_type=request.document_type,
            metadata=request.metadata
        )
        
        return IngestionResponse(
            document_id=document_id,
            status="success",
            message=f"Document '{request.title}' ingested successfully"
        )
        
    except Exception as e:
        logger.error(
            "ingestion_error",
            request_id=context.request_id,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/batch", response_model=List[IngestionResponse])
async def ingest_documents_batch(
    documents: List[IngestionRequest]
):
    """Ingest multiple documents in batch"""
    responses = []
    
    for doc in documents:
        try:
            document_id = await ingestion_service.ingest_document(
                title=doc.title,
                content=doc.content,
                source_url=doc.source_url,
                document_type=doc.document_type,
                metadata=doc.metadata
            )
            
            responses.append(IngestionResponse(
                document_id=document_id,
                status="success",
                message=f"Document '{doc.title}' ingested successfully"
            ))
        except Exception as e:
            responses.append(IngestionResponse(
                document_id=None,
                status="error",
                message=f"Failed to ingest '{doc.title}': {str(e)}"
            ))
    
    return responses

@router.post("/search", response_model=SearchResponse, dependencies=[Depends(check_rate_limit)])
async def search_documents(
    request: SearchRequest,
    context: RequestContext = Depends(get_request_context)
):
    """Search for relevant documents with optional reranking"""
    try:
        # Use cache if available (include reranking params in cache key)
        cache_key = f"search:{request.query}:{request.top_k}:{request.threshold}:{request.rerank}:{request.rerank_model}"
        
        if context.cache:
            cached_result = await context.cache.get(cache_key)
            if cached_result:
                logger.info("search_cache_hit", request_id=context.request_id)
                return SearchResponse.parse_raw(cached_result)
        
        # Perform search with reranking options
        results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filters,
            rerank=request.rerank,
            rerank_model=request.rerank_model,
            rerank_top_k=request.rerank_top_k
        )
        
        # Convert to response format
        chunks = [
            {
                "document_id": r.document_id,
                "document_title": r.document_title,
                "content": r.content,
                "chunk_index": r.chunk_index,
                "similarity_score": r.similarity_score,
                "rerank_score": r.rerank_score,
                "metadata": r.metadata
            }
            for r in results
        ]
        
        response = SearchResponse(
            query=request.query,
            results=chunks,
            total_results=len(chunks)
        )
        
        # Cache the result
        if context.cache:
            await context.cache.setex(
                cache_key,
                settings.cache_ttl,
                response.json()
            )
        
        return response
        
    except Exception as e:
        logger.error(
            "search_error",
            request_id=context.request_id,
            query=request.query,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(check_rate_limit)])
async def query_rag(
    request: QueryRequest,
    context: RequestContext = Depends(get_request_context)
):
    """Query the RAG system for an answer"""
    try:
        logger.info(
            "rag_query_start",
            request_id=context.request_id,
            query=request.query[:100]
        )
        
        # Retrieve relevant documents
        retrieved_chunks = await retriever.search(
            query=request.query,
            top_k=request.top_k or settings.retrieval_top_k,
            filters=request.filters
        )
        
        # Generate response
        response = await generator_service.generate(
            query=request.query,
            context_chunks=retrieved_chunks,
            model_name=request.model or settings.default_llm_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Prepare sources
        sources = [
            {
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "chunk_content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "similarity_score": chunk.similarity_score
            }
            for chunk in retrieved_chunks
        ]
        
        return QueryResponse(
            answer=response.text,
            sources=sources,
            metadata={
                "model_used": response.model_name,
                "tokens_used": response.token_count,
                "retrieval_count": len(retrieved_chunks),
                "request_id": context.request_id
            }
        )
        
    except Exception as e:
        logger.error(
            "query_error",
            request_id=context.request_id,
            query=request.query,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def stream_query(request: QueryRequest):
    """Stream the RAG response"""
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Stream retrieval status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Retrieving relevant documents...'})}\n\n"
            
            # Retrieve documents
            retrieved_chunks = await retriever.search(
                query=request.query,
                top_k=request.top_k or settings.retrieval_top_k,
                filters=request.filters
            )
            
            yield f"data: {json.dumps({'type': 'retrieved', 'count': len(retrieved_chunks), 'top_score': retrieved_chunks[0].similarity_score if retrieved_chunks else 0})}\n\n"
            
            # Stream generation
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            
            async for chunk in generator_service.stream_generate(
                query=request.query,
                context_chunks=retrieved_chunks,
                model_name=request.model or settings.default_llm_model
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
            # Send sources
            sources = [
                {
                    "document_title": chunk.document_title,
                    "similarity_score": chunk.similarity_score
                }
                for chunk in retrieved_chunks[:3]  # Top 3 sources
            ]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@router.put("/documents/{document_id}/embeddings")
async def update_document_embeddings(
    document_id: UUID,
    model_name: Optional[str] = None
):
    """Update embeddings for a specific document"""
    try:
        await ingestion_service.update_document_embeddings(
            document_id=document_id,
            model_name=model_name
        )
        
        return {
            "status": "success",
            "message": f"Embeddings updated for document {document_id}",
            "model_used": model_name or settings.default_embedding_model
        }
        
    except Exception as e:
        logger.error("embedding_update_error", document_id=str(document_id), error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(document_id: UUID):
    """Delete a document and all its chunks"""
    # TODO: Implement document deletion
    raise HTTPException(status_code=501, detail="Not implemented yet")

@router.get("/circuit-breakers", response_model=Dict[str, Any])
async def get_circuit_breakers_status():
    """Get status of all circuit breakers"""
    try:
        all_stats = CircuitBreaker.get_all_stats()
        
        # Add service-specific statuses
        embedding_status = embedding_service.get_circuit_breaker_status()
        generator_status = generator_service.get_circuit_breaker_status()
        
        return {
            "circuit_breakers": all_stats,
            "services": {
                "embeddings": embedding_status,
                "generator": generator_status
            },
            "summary": {
                "total_breakers": len(all_stats),
                "open_breakers": [
                    name for name, stats in all_stats.items() 
                    if stats["state"] == "open"
                ],
                "degraded_services": [
                    name for name, stats in all_stats.items() 
                    if stats["state"] in ["open", "half_open"]
                ]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get circuit breaker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/circuit-breakers/{breaker_name}", response_model=Dict[str, Any])
async def get_circuit_breaker_details(breaker_name: str):
    """Get detailed status of a specific circuit breaker"""
    breaker = CircuitBreaker.get_instance(breaker_name)
    
    if not breaker:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit breaker '{breaker_name}' not found"
        )
    
    return breaker.get_stats()

@router.post("/circuit-breakers/{breaker_name}/reset")
async def reset_circuit_breaker(breaker_name: str):
    """Manually reset a circuit breaker"""
    breaker = CircuitBreaker.get_instance(breaker_name)
    
    if not breaker:
        raise HTTPException(
            status_code=404,
            detail=f"Circuit breaker '{breaker_name}' not found"
        )
    
    breaker.reset()
    
    return {
        "status": "success",
        "message": f"Circuit breaker '{breaker_name}' has been reset",
        "current_state": breaker.state.value
    }

@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    db_healthy: bool = Depends(check_database_health),
    redis_healthy: bool = Depends(check_redis_health)
):
    """Detailed health check including circuit breaker status"""
    
    # Get circuit breaker states
    circuit_breaker_stats = CircuitBreaker.get_all_stats()
    
    # Determine overall health
    open_breakers = [
        name for name, stats in circuit_breaker_stats.items() 
        if stats["state"] == "open"
    ]
    
    if not db_healthy or open_breakers:
        overall_status = "unhealthy"
    elif not redis_healthy or any(
        stats["state"] == "half_open" 
        for stats in circuit_breaker_stats.values()
    ):
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": {
                "status": "healthy" if db_healthy else "unhealthy",
                "circuit_breaker": circuit_breaker_stats.get("database", {}).get("state", "unknown")
            },
            "redis": {
                "status": "healthy" if redis_healthy else "unhealthy",
                "circuit_breaker": circuit_breaker_stats.get("redis", {}).get("state", "unknown")
            },
            "openai": {
                "status": "healthy" if circuit_breaker_stats.get("openai", {}).get("state") != "open" else "unhealthy",
                "circuit_breaker": circuit_breaker_stats.get("openai", {}).get("state", "unknown")
            }
        },
        "models": {
            "embedding_model": settings.default_embedding_model,
            "llm_model": settings.default_llm_model
        },
        "circuit_breakers": {
            "total": len(circuit_breaker_stats),
            "open": len(open_breakers),
            "details": circuit_breaker_stats
        }
    }

@router.get("/reranking/status", response_model=RerankingStatusResponse)
async def get_reranking_status():
    """Get current reranking configuration and status"""
    try:
        return RerankingStatusResponse(
            enabled=retriever.reranking_enabled,
            default_model=retriever._default_reranking_model,
            available_models=retriever.get_reranking_models()
        )
    except Exception as e:
        logger.error(f"Failed to get reranking status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/reranking/config")
async def update_reranking_config(request: UpdateRerankingRequest):
    """Update reranking configuration"""
    try:
        retriever.enable_reranking(
            enabled=request.enabled,
            model=request.model
        )
        
        return {
            "status": "success",
            "message": f"Reranking {'enabled' if request.enabled else 'disabled'}",
            "model": retriever._default_reranking_model if request.enabled else None
        }
    except Exception as e:
        logger.error(f"Failed to update reranking config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reranking/models", response_model=Dict[str, RerankingModelInfo])
async def get_reranking_models():
    """Get available reranking models"""
    try:
        return retriever.get_reranking_models()
    except Exception as e:
        logger.error(f"Failed to get reranking models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/compare")
async def compare_search_methods(request: SearchRequest):
    """Compare search results with and without reranking"""
    try:
        # Search without reranking
        results_without = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filters,
            rerank=False
        )
        
        # Search with reranking
        results_with = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filters,
            rerank=True,
            rerank_model=request.rerank_model
        )
        
        # Calculate differences
        def get_doc_ids(results):
            return [r.document_id for r in results]
        
        ids_without = get_doc_ids(results_without)
        ids_with = get_doc_ids(results_with)
        
        # Find rank changes
        rank_changes = {}
        for i, doc_id in enumerate(ids_with):
            if doc_id in ids_without:
                old_rank = ids_without.index(doc_id)
                rank_changes[doc_id] = {
                    "old_rank": old_rank + 1,
                    "new_rank": i + 1,
                    "change": old_rank - i
                }
        
        return {
            "query": request.query,
            "without_reranking": [
                {
                    "rank": i + 1,
                    "document_id": r.document_id,
                    "title": r.document_title,
                    "score": r.similarity_score
                }
                for i, r in enumerate(results_without)
            ],
            "with_reranking": [
                {
                    "rank": i + 1,
                    "document_id": r.document_id,
                    "title": r.document_title,
                    "similarity_score": r.similarity_score,
                    "rerank_score": r.rerank_score
                }
                for i, r in enumerate(results_with)
            ],
            "rank_changes": rank_changes,
            "summary": {
                "total_documents": len(set(ids_without + ids_with)),
                "documents_reordered": len(rank_changes),
                "new_documents": len(set(ids_with) - set(ids_without)),
                "removed_documents": len(set(ids_without) - set(ids_with))
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to compare search methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/multi-model")
async def search_multi_model(
    query: str,
    model_names: List[str],
    top_k: Optional[int] = 5,
    aggregation: str = "union",
    rerank: bool = True,
    rerank_model: Optional[str] = None
):
    """Search using multiple embedding models with optional reranking"""
    try:
        results = await retriever.search_multi_model(
            query=query,
            model_names=model_names,
            top_k=top_k,
            aggregation=aggregation,
            rerank=rerank,
            rerank_model=rerank_model
        )
        
        return {
            "query": query,
            "models_used": model_names,
            "aggregation": aggregation,
            "reranked": rerank,
            "results": [
                {
                    "document_id": r.document_id,
                    "document_title": r.document_title,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "similarity_score": r.similarity_score,
                    "rerank_score": r.rerank_score
                }
                for r in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Multi-model search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))