# src/rag_system/core/dependencies.py
"""
Dependency injection for FastAPI application.
Provides reusable dependencies for database sessions, cache, auth, etc.
"""

from typing import AsyncGenerator, Optional, Annotated
from fastapi import Depends, HTTPException, Header, status
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from functools import lru_cache

from .config import settings
from ..db.pgvector import db
from ..utils.monitoring import logger


# Database Dependencies
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    Automatically handles session lifecycle.
    """
    async with db.get_session() as session:
        yield session


# Cache Dependencies
@lru_cache()
def get_redis_settings():
    """Get Redis settings (cached)"""
    return {
        "url": settings.redis_url,
        "decode_responses": True,
        "encoding": "utf-8",
    }


async def get_redis_client() -> AsyncGenerator[Optional[redis.Redis], None]:
    """
    Dependency to get Redis client.
    Returns None if Redis is not configured.
    """
    if not settings.redis_url:
        yield None
        return
    
    redis_client = None
    try:
        redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            encoding="utf-8"
        )
        # Test connection
        await redis_client.ping()
        yield redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        yield None
    finally:
        if redis_client:
            await redis_client.close()


# API Key Dependencies
async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None
) -> str:
    """
    Verify API key from header.
    This is a simple implementation - enhance for production use.
    """
    # Skip API key check if not configured
    if not hasattr(settings, 'api_key_required') or not settings.api_key_required:
        return "default"
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header missing"
        )
    
    # In production, validate against database or secure storage
    # This is a placeholder implementation
    valid_api_keys = getattr(settings, 'valid_api_keys', [])
    if x_api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return x_api_key


# Rate Limiting Dependencies
class RateLimiter:
    """Simple in-memory rate limiter (use Redis in production)"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit"""
        # This is a simplified implementation
        # In production, use Redis with sliding window
        from datetime import datetime, timedelta
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests = {
            k: v for k, v in self.requests.items() 
            if v > minute_ago
        }
        
        # Count requests for this key
        key_requests = [
            t for k, t in self.requests.items() 
            if k.startswith(key)
        ]
        
        if len(key_requests) >= self.requests_per_minute:
            return False
        
        # Add this request
        self.requests[f"{key}:{now.timestamp()}"] = now
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)


async def check_rate_limit(
    api_key: str = Depends(verify_api_key),
    redis_client: Optional[redis.Redis] = Depends(get_redis_client)
) -> None:
    """
    Check rate limit for the current request.
    Uses Redis if available, falls back to in-memory.
    """
    if redis_client:
        # Redis-based rate limiting
        key = f"rate_limit:{api_key}"
        try:
            current = await redis_client.incr(key)
            if current == 1:
                await redis_client.expire(key, 60)  # 1 minute window
            
            if current > 60:  # 60 requests per minute
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
        except redis.RedisError as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fall through to in-memory rate limiter
    
    # Fallback to in-memory rate limiting
    if not await rate_limiter.check_rate_limit(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


# Service Dependencies
def get_db():
    """
    Get database instance.
    Used for services that need direct database access.
    """
    return db


def get_cache(
    redis_client: Optional[redis.Redis] = Depends(get_redis_client)
) -> Optional[redis.Redis]:
    """
    Get cache client.
    Returns Redis client if available, None otherwise.
    """
    return redis_client


# Request Context Dependencies
async def get_request_id(
    x_request_id: Annotated[Optional[str], Header()] = None
) -> str:
    """Get or generate request ID for tracing"""
    if x_request_id:
        return x_request_id
    
    import uuid
    return str(uuid.uuid4())


class RequestContext:
    """Container for request-scoped data"""
    
    def __init__(
        self,
        request_id: str,
        api_key: str,
        db_session: AsyncSession,
        cache: Optional[redis.Redis] = None
    ):
        self.request_id = request_id
        self.api_key = api_key
        self.db_session = db_session
        self.cache = cache


async def get_request_context(
    request_id: str = Depends(get_request_id),
    api_key: str = Depends(verify_api_key),
    db_session: AsyncSession = Depends(get_db_session),
    cache: Optional[redis.Redis] = Depends(get_cache)
) -> RequestContext:
    """Get complete request context"""
    return RequestContext(
        request_id=request_id,
        api_key=api_key,
        db_session=db_session,
        cache=cache
    )


# Health Check Dependencies
async def check_database_health() -> bool:
    """Check if database is healthy"""
    try:
        async with db.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def check_redis_health(
    redis_client: Optional[redis.Redis] = Depends(get_redis_client)
) -> bool:
    """Check if Redis is healthy"""
    if not redis_client:
        return True  # Redis is optional
    
    try:
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


# Model Dependencies
def get_embedding_model_name(
    model: Optional[str] = None
) -> str:
    """Get embedding model name with fallback to default"""
    return model or settings.default_embedding_model


def get_llm_model_name(
    model: Optional[str] = None
) -> str:
    """Get LLM model name with fallback to default"""
    return model or settings.default_llm_model


# Pagination Dependencies
class PaginationParams:
    """Common pagination parameters"""
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 20,
        max_page_size: int = 100
    ):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), max_page_size)
        self.offset = (self.page - 1) * self.page_size


def get_pagination(
    page: int = 1,
    page_size: int = 20
) -> PaginationParams:
    """Get pagination parameters"""
    return PaginationParams(page=page, page_size=page_size)


# Export commonly used dependencies
__all__ = [
    "get_db_session",
    "get_redis_client",
    "verify_api_key",
    "check_rate_limit",
    "get_db",
    "get_cache",
    "get_request_id",
    "get_request_context",
    "check_database_health",
    "check_redis_health",
    "get_embedding_model_name",
    "get_llm_model_name",
    "get_pagination",
    "RequestContext",
    "PaginationParams",
]