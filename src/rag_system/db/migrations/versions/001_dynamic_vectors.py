# src/rag_system/db/migrations/versions/001_dynamic_vectors.py

"""Add dynamic vector dimensions support

Revision ID: 001_dynamic_vectors
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy import text
import uuid

# revision identifiers, used by Alembic.
revision = '001_dynamic_vectors'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create embedding_models table
    op.create_table('embedding_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('dimension', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Integer(), nullable=True, default=1),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name'),
        sa.CheckConstraint('dimension > 0', name='positive_dimension')
    )
    
    # Create index on model_name
    op.create_index('idx_embedding_models_name', 'embedding_models', ['model_name'])
    
    # Insert default embedding models
    conn = op.get_bind()
    
    # Get existing model names from chunk_embeddings
    existing_models = conn.execute(
        text("SELECT DISTINCT model_name FROM chunk_embeddings")
    ).fetchall()
    
    # Common embedding model dimensions
    model_dimensions = {
        'text-embedding-ada-002': 1536,
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
    }
    
    # Insert known models
    for model_name, dimension in model_dimensions.items():
        model_id = str(uuid.uuid4())
        conn.execute(
            text("""
                INSERT INTO embedding_models (id, model_name, dimension, is_active)
                VALUES (:id, :model_name, :dimension, 1)
                ON CONFLICT (model_name) DO NOTHING
            """),
            {"id": model_id, "model_name": model_name, "dimension": dimension}
        )
    
    # Insert any existing models not in our list (assume 1536 dimension)
    for (model_name,) in existing_models:
        if model_name and model_name not in model_dimensions:
            model_id = str(uuid.uuid4())
            conn.execute(
                text("""
                    INSERT INTO embedding_models (id, model_name, dimension, is_active)
                    VALUES (:id, :model_name, 1536, 1)
                    ON CONFLICT (model_name) DO NOTHING
                """),
                {"id": model_id, "model_name": model_name}
            )
    
    # Create new chunk_embeddings_v2 table without vector column initially
    op.create_table('chunk_embeddings_v2',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('embedding_version', sa.Integer(), nullable=True, default=1),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['chunk_id'], ['document_chunks.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['model_id'], ['embedding_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Add vector columns for each dimension we need
    dimensions_needed = set()
    
    # Get all unique dimensions
    result = conn.execute(
        text("""
            SELECT DISTINCT em.dimension 
            FROM embedding_models em
            WHERE EXISTS (
                SELECT 1 FROM chunk_embeddings ce 
                WHERE ce.model_name = em.model_name
            )
        """)
    )
    
    for (dimension,) in result:
        dimensions_needed.add(dimension)
    
    # Add default dimension if no embeddings exist yet
    if not dimensions_needed:
        dimensions_needed.add(1536)
    
    # Create vector columns for each dimension
    for dimension in dimensions_needed:
        column_name = f'embedding_{dimension}'
        conn.execute(
            text(f"""
                ALTER TABLE chunk_embeddings_v2 
                ADD COLUMN {column_name} vector({dimension})
            """)
        )
    
    # Migrate data from old table to new table
    conn.execute(
        text("""
            INSERT INTO chunk_embeddings_v2 (id, chunk_id, model_id, embedding_version, embedding_1536, created_at)
            SELECT 
                ce.id,
                ce.chunk_id,
                em.id as model_id,
                ce.embedding_version,
                ce.embedding,
                ce.created_at
            FROM chunk_embeddings ce
            JOIN embedding_models em ON ce.model_name = em.model_name
            WHERE em.dimension = 1536
        """)
    )
    
    # Create indexes on new table
    op.create_index('idx_chunk_embeddings_v2_chunk_model', 
                    'chunk_embeddings_v2', 
                    ['chunk_id', 'model_id'], 
                    unique=True)
    
    # For each dimension, create HNSW index
    for dimension in dimensions_needed:
        column_name = f'embedding_{dimension}'
        index_name = f'idx_embeddings_v2_hnsw_{dimension}'
        conn.execute(
            text(f"""
                CREATE INDEX {index_name} ON chunk_embeddings_v2 
                USING hnsw ({column_name} vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64)
            """)
        )
    
    # Drop old table and rename new one
    op.drop_table('chunk_embeddings')
    op.rename_table('chunk_embeddings_v2', 'chunk_embeddings')
    
    # Update index names
    for dimension in dimensions_needed:
        old_name = f'idx_embeddings_v2_hnsw_{dimension}'
        new_name = f'idx_embeddings_hnsw_{dimension}'
        op.execute(f'ALTER INDEX {old_name} RENAME TO {new_name}')
    
    op.execute('ALTER INDEX idx_chunk_embeddings_v2_chunk_model RENAME TO idx_chunk_embeddings_chunk_model')


def downgrade():
    # This is a complex migration, downgrade would lose data
    # In production, you might want to implement a more careful downgrade
    raise NotImplementedError("Downgrade not supported for this migration")