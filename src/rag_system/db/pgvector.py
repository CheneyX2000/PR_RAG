# src/rag_system/db/pgvector.py
import asyncpg
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, select
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from uuid import UUID
import uuid

from ..core.config import settings
from .models import Base, EmbeddingModel, ChunkEmbedding
from ..utils.exceptions import DatabaseError, ValidationError
from ..utils.monitoring import logger
from ..utils.circuit_breaker import CircuitBreakers, CircuitBreakerError

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
        self._dimension_cache = {}  # Cache model dimensions
    
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
        
        # Load dimension cache
        await self._load_dimension_cache()
    
    async def _init_database(self):
        """Create tables and initialize pgvector extension"""
        async with self.engine.begin() as conn:
            # Create pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
    
    async def _load_dimension_cache(self):
        """Load embedding model dimensions into cache"""
        async with self.get_session() as session:
            result = await session.execute(
                select(EmbeddingModel.model_name, EmbeddingModel.dimension)
            )
            for model_name, dimension in result:
                self._dimension_cache[model_name] = dimension
    
    async def ensure_embedding_model(self, model_name: str, dimension: int) -> UUID:
        """Ensure an embedding model exists and return its ID"""
        async with self.get_session() as session:
            # Check if model exists
            result = await session.execute(
                select(EmbeddingModel).where(EmbeddingModel.model_name == model_name)
            )
            model = result.scalar_one_or_none()
            
            if model:
                if model.dimension != dimension:
                    raise ValidationError(
                        f"Model {model_name} exists with dimension {model.dimension}, "
                        f"but dimension {dimension} was requested"
                    )
                return model.id
            
            # Create new model
            model = EmbeddingModel(
                model_name=model_name,
                dimension=dimension,
                is_active=True
            )
            session.add(model)
            await session.commit()
            
            # Update cache
            self._dimension_cache[model_name] = dimension
            
            # Ensure vector column exists
            await self._ensure_vector_column(dimension)
            
            return model.id
    
    async def _ensure_vector_column(self, dimension: int):
        """Ensure a vector column exists for the given dimension"""
        column_name = f'embedding_{dimension}'
        
        async with self.acquire() as conn:
            # Check if column exists
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name = 'chunk_embeddings' 
                    AND column_name = $1
                )
            """, column_name)
            
            if not result:
                logger.info(f"Creating vector column for dimension {dimension}")
                
                # Add the column
                await conn.execute(f"""
                    ALTER TABLE chunk_embeddings 
                    ADD COLUMN {column_name} vector({dimension})
                """)
                
                # Create HNSW index
                index_name = f'idx_embeddings_hnsw_{dimension}'
                await conn.execute(f"""
                    CREATE INDEX {index_name} ON chunk_embeddings 
                    USING hnsw ({column_name} vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64)
                """)
    
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
    
    async def _execute_insert(self, conn, query: str, *args):
        """Execute insert query"""
        return await conn.fetchval(query, *args)
    
    async def insert_embedding(
        self,
        chunk_id: UUID,
        model_name: str,
        embedding: List[float],
        dimension: int,
        version: int = 1
    ) -> UUID:
        """Insert an embedding with proper dimension handling and circuit breaker protection"""
        # Ensure model exists
        model_id = await self.ensure_embedding_model(model_name, dimension)
        
        column_name = f'embedding_{dimension}'
        embedding_str = f"[{','.join(map(str, embedding))}]"
        
        try:
            async with self.acquire() as conn:
                # Check if embedding already exists
                existing = await CircuitBreakers.database.call_async(
                    conn.fetchval,
                    """
                    SELECT id FROM chunk_embeddings 
                    WHERE chunk_id = $1 AND model_id = $2
                    """, 
                    chunk_id, 
                    model_id
                )
                
                if existing:
                    # Update existing
                    query = f"""
                        UPDATE chunk_embeddings 
                        SET {column_name} = $1::vector,
                            embedding_version = $2
                        WHERE id = $3
                        RETURNING id
                    """
                    result = await CircuitBreakers.database.call_async(
                        self._execute_insert,
                        conn,
                        query,
                        embedding_str,
                        version,
                        existing
                    )
                else:
                    # Insert new
                    query = f"""
                        INSERT INTO chunk_embeddings 
                        (chunk_id, model_id, embedding_version, {column_name})
                        VALUES ($1, $2, $3, $4::vector)
                        RETURNING id
                    """
                    result = await CircuitBreakers.database.call_async(
                        self._execute_insert,
                        conn,
                        query,
                        chunk_id,
                        model_id,
                        version,
                        embedding_str
                    )
                
                return result
                
        except CircuitBreakerError as e:
            logger.error(f"Database circuit breaker open during insert: {e}")
            raise DatabaseError(
                "Database is temporarily unavailable. Please try again later."
            )
        except Exception as e:
            logger.error(f"Embedding insert error: {e}")
            raise DatabaseError(f"Failed to insert embedding: {str(e)}")
    
    async def _execute_similarity_search(self, conn, query: str, *args):
        """Execute similarity search query"""
        return await conn.fetch(query, *args)
    
    async def similarity_search(
        self, 
        embedding: List[float], 
        model_name: str,
        top_k: int = 5,
        threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search with dynamic dimensions and circuit breaker protection"""
        # Get dimension from cache or database
        if model_name not in self._dimension_cache:
            await self._load_dimension_cache()
        
        dimension = self._dimension_cache.get(model_name)
        if not dimension:
            raise ValidationError(f"Unknown embedding model: {model_name}")
        
        if len(embedding) != dimension:
            raise ValidationError(
                f"Embedding dimension {len(embedding)} doesn't match "
                f"model {model_name} dimension {dimension}"
            )
        
        column_name = f'embedding_{dimension}'
        embedding_str = f"[{','.join(map(str, embedding))}]"
        
        try:
            async with self.acquire() as conn:
                # Build the query dynamically
                query = f"""
                    SELECT 
                        d.id::text as document_id,
                        d.title,
                        dc.id::text as chunk_id,
                        dc.content,
                        dc.chunk_index,
                        1 - (ce.{column_name} <=> $1::vector) as similarity,
                        d.metadata
                    FROM chunk_embeddings ce
                    JOIN embedding_models em ON ce.model_id = em.id
                    JOIN document_chunks dc ON ce.chunk_id = dc.id
                    JOIN documents d ON dc.document_id = d.id
                    WHERE em.model_name = $2
                        AND ce.{column_name} IS NOT NULL
                        AND 1 - (ce.{column_name} <=> $1::vector) > $3
                        AND ($4::jsonb IS NULL OR d.metadata @> $4::jsonb)
                    ORDER BY ce.{column_name} <=> $1::vector
                    LIMIT $5
                """
                
                # Use circuit breaker for database query
                rows = await CircuitBreakers.database.call_async(
                    self._execute_similarity_search,
                    conn,
                    query,
                    embedding_str,
                    model_name,
                    threshold,
                    filters,
                    top_k
                )
                
                return [dict(row) for row in rows]
                
        except CircuitBreakerError as e:
            logger.error(f"Database circuit breaker open: {e}")
            raise DatabaseError(
                "Database is temporarily unavailable. Please try again later."
            )
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            raise DatabaseError(f"Failed to perform similarity search: {str(e)}")
    
    async def get_supported_dimensions(self) -> List[int]:
        """Get list of dimensions that have vector columns"""
        async with self.acquire() as conn:
            result = await conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'chunk_embeddings' 
                AND column_name LIKE 'embedding_%'
            """)
            
            dimensions = []
            for row in result:
                column_name = row['column_name']
                # Extract dimension from column name
                try:
                    dimension = int(column_name.split('_')[1])
                    dimensions.append(dimension)
                except (IndexError, ValueError):
                    continue
            
            return sorted(dimensions)
    
    async def migrate_embeddings_to_dimension(
        self,
        source_model: str,
        target_model: str,
        target_dimension: int,
        embeddings_map: Dict[str, List[float]]
    ):
        """Migrate embeddings from one model/dimension to another"""
        # This is a placeholder for the migration logic
        # In practice, you would re-embed the chunks with the new model
        logger.info(
            f"Migrating embeddings from {source_model} to {target_model} "
            f"with dimension {target_dimension}"
        )
        
        # Ensure target model exists
        target_model_id = await self.ensure_embedding_model(target_model, target_dimension)
        
        # The actual migration would involve:
        # 1. Fetching all chunks that have embeddings for source_model
        # 2. Re-embedding them with the target model
        # 3. Inserting the new embeddings
        
        # This is a complex operation that should be done in batches
        # and possibly as a background job
    
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