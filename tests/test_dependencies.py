# tests/test_dependencies.py
"""
Comprehensive tests for dependency injection module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

from src.rag_system.core.dependencies import (
    get_db_session,
    get_redis_client,
    verify_api_key,
    check_rate_limit,
    get_request_id,
    get_request_context,
    check_database_health,
    check_redis_health,
    get_embedding_model_name,
    get_llm_model_name,
    get_pagination,
    RateLimiter,
    RequestContext,
    PaginationParams,
    rate_limiter
)


class TestDatabaseDependencies:
    """Test database-related dependencies"""
    
    @pytest.mark.asyncio
    async def test_get_db_session(self):
        """Test database session dependency"""
        mock_session = AsyncMock(spec=AsyncSession)
        
        with patch('src.rag_system.db.pgvector.db.get_session') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_session
            
            async for session in get_db_session():
                assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_check_database_health_success(self):
        """Test database health check when healthy"""
        with patch('src.rag_system.db.pgvector.db.acquire') as mock_acquire:
            mock_conn = AsyncMock()
            mock_conn.fetchval.return_value = 1
            mock_acquire.return_value.__aenter__.return_value = mock_conn
            
            result = await check_database_health()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_database_health_failure(self):
        """Test database health check when unhealthy"""
        with patch('src.rag_system.db.pgvector.db.acquire') as mock_acquire:
            mock_acquire.side_effect = Exception("Connection failed")
            
            result = await check_database_health()
            assert result is False


class TestCacheDependencies:
    """Test cache-related dependencies"""
    
    @pytest.mark.asyncio
    async def test_get_redis_client_success(self):
        """Test Redis client when configured"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379"
            
            with patch('redis.asyncio.from_url') as mock_redis:
                mock_client = AsyncMock()
                mock_client.ping = AsyncMock()
                mock_redis.return_value = mock_client
                
                async for client in get_redis_client():
                    assert client == mock_client
                
                mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_redis_client_not_configured(self):
        """Test Redis client when not configured"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.redis_url = None
            
            async for client in get_redis_client():
                assert client is None
    
    @pytest.mark.asyncio
    async def test_get_redis_client_connection_failure(self):
        """Test Redis client when connection fails"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.redis_url = "redis://localhost:6379"
            
            with patch('redis.asyncio.from_url') as mock_redis:
                mock_client = AsyncMock()
                mock_client.ping.side_effect = Exception("Connection refused")
                mock_redis.return_value = mock_client
                
                async for client in get_redis_client():
                    assert client is None
    
    @pytest.mark.asyncio
    async def test_check_redis_health_success(self):
        """Test Redis health check when healthy"""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock()
        
        result = await check_redis_health(redis_client=mock_client)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_redis_health_not_configured(self):
        """Test Redis health check when not configured"""
        result = await check_redis_health(redis_client=None)
        assert result is True  # Redis is optional
    
    @pytest.mark.asyncio
    async def test_check_redis_health_failure(self):
        """Test Redis health check when unhealthy"""
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Ping failed")
        
        result = await check_redis_health(redis_client=mock_client)
        assert result is False


class TestAuthDependencies:
    """Test authentication dependencies"""
    
    @pytest.mark.asyncio
    async def test_verify_api_key_not_required(self):
        """Test API key verification when not required"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            delattr(mock_settings, 'api_key_required')  # Not configured
            
            result = await verify_api_key(x_api_key=None)
            assert result == "default"
    
    @pytest.mark.asyncio
    async def test_verify_api_key_missing(self):
        """Test API key verification when key is missing"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.api_key_required = True
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(x_api_key=None)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "X-API-Key header missing" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_verify_api_key_invalid(self):
        """Test API key verification with invalid key"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.api_key_required = True
            mock_settings.valid_api_keys = ["valid-key-1", "valid-key-2"]
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(x_api_key="invalid-key")
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "Invalid API key" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_verify_api_key_valid(self):
        """Test API key verification with valid key"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.api_key_required = True
            mock_settings.valid_api_keys = ["valid-key-1", "valid-key-2"]
            
            result = await verify_api_key(x_api_key="valid-key-1")
            assert result == "valid-key-1"


class TestRateLimiting:
    """Test rate limiting dependencies"""
    
    def test_rate_limiter_initialization(self):
        """Test RateLimiter initialization"""
        limiter = RateLimiter(requests_per_minute=30)
        assert limiter.requests_per_minute == 30
        assert limiter.requests == {}
    
    @pytest.mark.asyncio
    async def test_rate_limiter_within_limit(self):
        """Test rate limiter when within limit"""
        limiter = RateLimiter(requests_per_minute=5)
        
        # Make requests within limit
        for i in range(5):
            result = await limiter.check_rate_limit(f"user-{i}")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_exceeds_limit(self):
        """Test rate limiter when exceeding limit"""
        limiter = RateLimiter(requests_per_minute=2)
        
        # Make requests exceeding limit
        assert await limiter.check_rate_limit("user") is True
        assert await limiter.check_rate_limit("user") is True
        assert await limiter.check_rate_limit("user") is False
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_with_redis(self):
        """Test rate limiting with Redis"""
        mock_redis = AsyncMock()
        mock_redis.incr.return_value = 1
        mock_redis.expire = AsyncMock()
        
        await check_rate_limit(api_key="test-key", redis_client=mock_redis)
        
        mock_redis.incr.assert_called_once_with("rate_limit:test-key")
        mock_redis.expire.assert_called_once_with("rate_limit:test-key", 60)
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_redis_exceeded(self):
        """Test rate limiting exceeded with Redis"""
        mock_redis = AsyncMock()
        mock_redis.incr.return_value = 61  # Over limit
        
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(api_key="test-key", redis_client=mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_fallback(self):
        """Test rate limiting fallback when Redis fails"""
        mock_redis = AsyncMock()
        mock_redis.incr.side_effect = redis.RedisError("Connection failed")
        
        # Should fall back to in-memory rate limiter
        with patch.object(rate_limiter, 'check_rate_limit', return_value=True):
            await check_rate_limit(api_key="test-key", redis_client=mock_redis)


class TestRequestContext:
    """Test request context dependencies"""
    
    @pytest.mark.asyncio
    async def test_get_request_id_provided(self):
        """Test request ID when provided in header"""
        request_id = "custom-request-id"
        result = await get_request_id(x_request_id=request_id)
        assert result == request_id
    
    @pytest.mark.asyncio
    async def test_get_request_id_generated(self):
        """Test request ID generation when not provided"""
        result = await get_request_id(x_request_id=None)
        assert len(result) == 36  # UUID format
        assert "-" in result
    
    @pytest.mark.asyncio
    async def test_get_request_context(self):
        """Test complete request context creation"""
        mock_session = AsyncMock()
        mock_redis = AsyncMock()
        
        with patch('src.rag_system.core.dependencies.get_request_id', return_value="req-123"):
            with patch('src.rag_system.core.dependencies.verify_api_key', return_value="api-key"):
                ctx = await get_request_context(
                    request_id="req-123",
                    api_key="api-key",
                    db_session=mock_session,
                    cache=mock_redis
                )
                
                assert isinstance(ctx, RequestContext)
                assert ctx.request_id == "req-123"
                assert ctx.api_key == "api-key"
                assert ctx.db_session == mock_session
                assert ctx.cache == mock_redis


class TestModelDependencies:
    """Test model-related dependencies"""
    
    def test_get_embedding_model_name_default(self):
        """Test getting default embedding model name"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.default_embedding_model = "text-embedding-ada-002"
            
            result = get_embedding_model_name(model=None)
            assert result == "text-embedding-ada-002"
    
    def test_get_embedding_model_name_specified(self):
        """Test getting specified embedding model name"""
        result = get_embedding_model_name(model="custom-model")
        assert result == "custom-model"
    
    def test_get_llm_model_name_default(self):
        """Test getting default LLM model name"""
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.default_llm_model = "gpt-4o-mini"
            
            result = get_llm_model_name(model=None)
            assert result == "gpt-4o-mini"
    
    def test_get_llm_model_name_specified(self):
        """Test getting specified LLM model name"""
        result = get_llm_model_name(model="gpt-4o")
        assert result == "gpt-4o"


class TestPaginationDependencies:
    """Test pagination dependencies"""
    
    def test_pagination_params_defaults(self):
        """Test PaginationParams with defaults"""
        params = PaginationParams()
        assert params.page == 1
        assert params.page_size == 20
        assert params.offset == 0
    
    def test_pagination_params_custom(self):
        """Test PaginationParams with custom values"""
        params = PaginationParams(page=3, page_size=50)
        assert params.page == 3
        assert params.page_size == 50
        assert params.offset == 100  # (3-1) * 50
    
    def test_pagination_params_constraints(self):
        """Test PaginationParams constraints"""
        # Test minimum page
        params = PaginationParams(page=0)
        assert params.page == 1
        
        # Test maximum page size
        params = PaginationParams(page_size=200, max_page_size=100)
        assert params.page_size == 100
        
        # Test minimum page size
        params = PaginationParams(page_size=0)
        assert params.page_size == 1
    
    def test_get_pagination(self):
        """Test get_pagination dependency"""
        params = get_pagination(page=5, page_size=25)
        assert isinstance(params, PaginationParams)
        assert params.page == 5
        assert params.page_size == 25
        assert params.offset == 100