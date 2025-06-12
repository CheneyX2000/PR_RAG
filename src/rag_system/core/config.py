# src/rag_system/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any

class Settings(BaseSettings):
    # Application
    app_name: str = "RAG System"
    debug: bool = False
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/ragdb"
    pgvector_extension: bool = True
    
    # Models
    default_embedding_model: str = "text-embedding-ada-002"
    default_llm_model: str = "gpt-4o-mini"
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Performance
    max_chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    
    # Redis (optional for now)
    redis_url: Optional[str] = None
    cache_ttl: int = 3600
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

settings = Settings()