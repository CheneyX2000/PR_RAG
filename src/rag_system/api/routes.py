# src/rag_system/api/routes.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
import json
from uuid import UUID

from .schemas import (
    QueryRequest,
    QueryResponse,
    DocumentRequest,
    DocumentResponse,
    IngestionRequest,
    IngestionResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse
)
from ..services.retriever import retriever
from ..services.ingestion import ingestion_service
from ..services.generator import generator_service
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
    status = "healthy" if db_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        embedding_model=settings.default_embedding_model,
        llm_model=settings.default_llm_model
    )


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
    """Search for relevant documents"""
    try:
        # Use cache if available
        cache_key = f"search:{request.query}:{request.top_k}:{request.threshold}"
        
        if context.cache:
            cached_result = await context.cache.get(cache_key)
            if cached_result:
                logger.info("search_cache_hit", request_id=context.request_id)
                return SearchResponse.parse_raw(cached_result)
        
        # Perform search
        results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filters
        )
        
        # Convert to response format
        chunks = [
            {
                "document_id": r.document_id,
                "document_title": r.document_title,
                "content": r.content,
                "chunk_index": r.chunk_index,
                "similarity_score": r.similarity_score,
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