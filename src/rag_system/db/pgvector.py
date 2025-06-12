# src/rag_system/db/pgvector.py
import asyncpg
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import numpy as np

from ..core.config import settings
from .models import Base

class PgVectorDB:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        # Convert to async URL
        if self.database_url.startswith("postgresql://"):
            self.async_database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        else:
            self.async_database_url = self.database_url
            
        self.engine = None
        self.async_session_maker = None
        self._pool = None
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        # Create SQLAlchemy engine
        self.engine = create_async_engine(
            self.async_database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=20
        )
        
        # Create session maker
        self.async_session_maker = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create connection pool for raw queries
        self._pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Initialize database
        await self._init_database()
    
    async def _init_database(self):
        """Create tables and initialize pgvector extension"""
        async with self.engine.begin() as conn:
            # Create pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Create indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
                ON chunk_embeddings 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
            """))
    
    @asynccontextmanager
    async def get_session(self):
        """Get SQLAlchemy session"""
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def acquire(self):
        """Get raw connection from pool"""
        async with self._pool.acquire() as conn:
            yield conn
    
    async def similarity_search(
        self, 
        embedding: List[float], 
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        embedding_str = f"[{','.join(map(str, embedding))}]"
        
        async with self.acquire() as conn:
            query = """
                SELECT 
                    d.id::text as document_id,
                    d.title,
                    dc.id::text as chunk_id,
                    dc.content,
                    dc.chunk_index,
                    1 - (ce.embedding <=> $1::vector) as similarity
                FROM chunk_embeddings ce
                JOIN document_chunks dc ON ce.chunk_id = dc.id
                JOIN documents d ON dc.document_id = d.id
                WHERE 1 - (ce.embedding <=> $1::vector) > $2
                    AND ($3::jsonb IS NULL OR d.metadata @> $3::jsonb)
                ORDER BY ce.embedding <=> $1::vector
                LIMIT $4
            """
            
            rows = await conn.fetch(
                query, 
                embedding_str, 
                threshold,
                filters,
                top_k
            )
            
            return [dict(row) for row in rows]
    
    async def close(self):
        """Close all connections"""
        if self._pool:
            await self._pool.close()
        if self.engine:
            await self.engine.dispose()

# Global instance
db = PgVectorDB()

# Helper function to initialize database
async def init_database():
    await db.initialize()
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_database())