# tests/test_middleware.py
"""
Comprehensive tests for API middleware.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
import time
from uuid import uuid4

from fastapi import Request, Response
from starlette.types import ASGIApp

from src.rag_system.api.middleware import (
    TimingMiddleware,
    ErrorHandlingMiddleware,
    SecurityHeadersMiddleware,
    setup_middleware
)


class TestTimingMiddleware:
    """Test cases for timing middleware"""
    
    @pytest.fixture
    def middleware(self):
        """Create timing middleware instance"""
        return TimingMiddleware(app=Mock())
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/v1/test"
        request.headers = {"X-Request-ID": str(uuid4())}
        return request
    
    @pytest.mark.asyncio
    async def test_timing_middleware_adds_headers(self, middleware, mock_request):
        """Test that timing middleware adds appropriate headers"""
        response = Response(content="test", status_code=200)
        
        async def call_next(request):
            await asyncio.sleep(0.1)  # Simulate processing time
            return response
        
        with patch('src.rag_system.utils.monitoring.logger') as mock_logger:
            result = await middleware.dispatch(mock_request, call_next)
            
            # Check headers were added
            assert "X-Request-ID" in result.headers
            assert "X-Response-Time" in result.headers
            
            # Check response time is reasonable
            response_time = float(result.headers["X-Response-Time"])
            assert 0.09 < response_time < 0.2  # Should be around 0.1s
            
            # Check logging
            assert mock_logger.info.call_count == 2  # start and complete
    
    @pytest.mark.asyncio
    async def test_timing_middleware_generates_request_id(self, middleware):
        """Test request ID generation when not provided"""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/v1/test"
        request.headers = {}  # No X-Request-ID
        
        response = Response(content="test", status_code=201)
        
        async def call_next(request):
            return response
        
        result = await middleware.dispatch(request, call_next)
        
        # Should generate a request ID
        assert "X-Request-ID" in result.headers
        assert len(result.headers["X-Request-ID"]) == 36  # UUID length
    
    @pytest.mark.asyncio
    async def test_timing_middleware_metrics(self, middleware, mock_request):
        """Test that metrics are recorded"""
        response = Response(content="test", status_code=200)
        
        async def call_next(request):
            return response
        
        with patch('src.rag_system.utils.monitoring.query_duration') as mock_metric:
            await middleware.dispatch(mock_request, call_next)
            
            # Check metrics were recorded
            mock_metric.labels.assert_called_once_with(
                method="GET",
                endpoint="/api/v1/test",
                status=200
            )


class TestErrorHandlingMiddleware:
    """Test cases for error handling middleware"""
    
    @pytest.fixture
    def middleware(self):
        """Create error handling middleware instance"""
        return ErrorHandlingMiddleware(app=Mock())
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/v1/test"
        request.headers = {"X-Request-ID": "test-id"}
        return request
    
    @pytest.mark.asyncio
    async def test_error_handling_success(self, middleware, mock_request):
        """Test middleware passes through successful responses"""
        response = Response(content="success", status_code=200)
        
        async def call_next(request):
            return response
        
        result = await middleware.dispatch(mock_request, call_next)
        assert result == response
    
    @pytest.mark.asyncio
    async def test_error_handling_exception(self, middleware, mock_request):
        """Test middleware handles exceptions"""
        async def call_next(request):
            raise ValueError("Test error")
        
        with patch('src.rag_system.utils.monitoring.logger') as mock_logger:
            with patch('src.rag_system.core.config.settings') as mock_settings:
                mock_settings.debug = False
                
                result = await middleware.dispatch(mock_request, call_next)
                
                # Check error response
                assert result.status_code == 500
                body = json.loads(result.body)
                assert body["error"] == "Internal server error"
                assert body["detail"] == "An unexpected error occurred"
                assert body["request_id"] == "test-id"
                
                # Check logging
                mock_logger.error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_debug_mode(self, middleware, mock_request):
        """Test error handling in debug mode shows details"""
        async def call_next(request):
            raise RuntimeError("Detailed error message")
        
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.debug = True
            
            result = await middleware.dispatch(mock_request, call_next)
            
            body = json.loads(result.body)
            assert body["detail"] == "Detailed error message"


class TestSecurityHeadersMiddleware:
    """Test cases for security headers middleware"""
    
    @pytest.fixture
    def middleware(self):
        """Create security headers middleware instance"""
        return SecurityHeadersMiddleware(app=Mock())
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        return Mock(spec=Request)
    
    @pytest.mark.asyncio
    async def test_security_headers_added(self, middleware, mock_request):
        """Test that security headers are added"""
        response = Response(content="test", status_code=200)
        
        async def call_next(request):
            return response
        
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.debug = False
            
            result = await middleware.dispatch(mock_request, call_next)
            
            # Check security headers
            assert result.headers["X-Content-Type-Options"] == "nosniff"
            assert result.headers["X-Frame-Options"] == "DENY"
            assert result.headers["X-XSS-Protection"] == "1; mode=block"
            assert result.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
            assert "Content-Security-Policy" in result.headers
    
    @pytest.mark.asyncio
    async def test_security_headers_debug_mode(self, middleware, mock_request):
        """Test CSP header is not added in debug mode"""
        response = Response(content="test", status_code=200)
        
        async def call_next(request):
            return response
        
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.debug = True
            
            result = await middleware.dispatch(mock_request, call_next)
            
            # CSP should not be added in debug mode
            assert "Content-Security-Policy" not in result.headers
            # But other security headers should still be present
            assert "X-Content-Type-Options" in result.headers


class TestSetupMiddleware:
    """Test cases for middleware setup"""
    
    def test_setup_middleware_cors(self):
        """Test CORS middleware setup"""
        app = Mock()
        
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.allowed_origins = "http://localhost:3000,http://localhost:8000"
            mock_settings.debug = False
            
            setup_middleware(app)
            
            # Check CORS middleware was added
            cors_call = None
            for call in app.add_middleware.call_args_list:
                if "CORSMiddleware" in str(call):
                    cors_call = call
                    break
            
            assert cors_call is not None
            kwargs = cors_call[1]
            assert kwargs["allow_origins"] == ["http://localhost:3000", "http://localhost:8000"]
            assert kwargs["allow_credentials"] is True
            assert kwargs["allow_methods"] == ["*"]
            assert kwargs["allow_headers"] == ["*"]
    
    def test_setup_middleware_trusted_host(self):
        """Test trusted host middleware setup"""
        app = Mock()
        
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.debug = False
            mock_settings.allowed_hosts = ["example.com", "*.example.com"]
            
            setup_middleware(app)
            
            # Check if TrustedHostMiddleware was added
            trusted_host_call = None
            for call in app.add_middleware.call_args_list:
                if "TrustedHostMiddleware" in str(call):
                    trusted_host_call = call
                    break
            
            assert trusted_host_call is not None
    
    def test_setup_middleware_custom_middleware(self):
        """Test custom middleware are added in correct order"""
        app = Mock()
        
        with patch('src.rag_system.core.config.settings') as mock_settings:
            mock_settings.debug = True
            
            setup_middleware(app)
            
            # Check custom middleware were added
            middleware_names = []
            for call in app.add_middleware.call_args_list:
                middleware_names.append(str(call[0][0]))
            
            # Order matters - executed in reverse
            assert "ErrorHandlingMiddleware" in str(middleware_names)
            assert "SecurityHeadersMiddleware" in str(middleware_names)
            assert "TimingMiddleware" in str(middleware_names)


import asyncio