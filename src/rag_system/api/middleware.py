# src/rag_system/api/middleware.py
"""
Middleware for the FastAPI application.
Handles CORS, logging, timing, and error handling.
"""

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import uuid
from typing import Callable
import json

from ..core.config import settings
from ..utils.monitoring import logger, query_duration


class TimingMiddleware(BaseHTTPMiddleware):
    """Add request timing information to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Add request ID if not present
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Log request
        logger.info(
            "http_request_start",
            method=request.method,
            path=request.url.path,
            request_id=request_id
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add timing headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}"
        
        # Log response
        logger.info(
            "http_request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            request_id=request_id
        )
        
        # Record metrics
        with query_duration.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).time():
            pass
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(
                "unhandled_error",
                error=str(e),
                method=request.method,
                path=request.url.path,
                traceback=True
            )
            
            # Return a generic error response
            return Response(
                content=json.dumps({
                    "error": "Internal server error",
                    "detail": str(e) if settings.debug else "An unexpected error occurred",
                    "request_id": request.headers.get("X-Request-ID", "unknown")
                }),
                status_code=500,
                media_type="application/json"
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add CSP if not in debug mode
        if not settings.debug:
            response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


def setup_middleware(app: ASGIApp):
    """Configure all middleware for the application"""
    
    # CORS middleware
    allowed_origins = getattr(settings, 'allowed_origins', ["*"])
    if isinstance(allowed_origins, str):
        allowed_origins = [origin.strip() for origin in allowed_origins.split(",")]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )
    
    # Trusted host middleware (production only)
    if not settings.debug:
        allowed_hosts = getattr(settings, 'allowed_hosts', ["*"])
        if allowed_hosts != ["*"]:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )
    
    # Custom middleware (order matters - executed in reverse order)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TimingMiddleware)
    
    logger.info(
        "middleware_configured",
        cors_origins=allowed_origins,
        debug_mode=settings.debug
    )