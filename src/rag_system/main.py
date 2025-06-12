# src/rag_system/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from .api.routes import router
from .api.middleware import setup_middleware
from .db.pgvector import db
from .core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting RAG System...")
    await db.initialize()
    logger.info("Database initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG System...")
    await db.close()
    logger.info("Cleanup completed")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug else None
)

# Setup middleware
setup_middleware(app)

# Include routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "docs": "/docs" if settings.debug else "Disabled in production"
    }

# Prometheus metrics endpoint (optional)
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_system.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )