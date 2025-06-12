# docs/README_migrations.md
# Dynamic Vector Dimensions Migration Guide

This guide explains how to migrate your RAG system to support dynamic vector dimensions, allowing you to use different embedding models with varying dimensions (384, 768, 1536, 3072, etc.).

## 🚨 Important: Backup Your Data

Before running any migrations, **backup your database**:

```bash
pg_dump -h localhost -U user -d ragdb > ragdb_backup_$(date +%Y%m%d_%H%M%S).sql
```

## 📋 Migration Overview

The migration transforms the fixed-dimension vector storage to a dynamic system that:
- Supports multiple embedding models with different dimensions
- Allows hot-switching between models
- Maintains backward compatibility with existing embeddings
- Enables gradual migration to new models

## 🚀 Quick Start

### 1. Install Dependencies

First, ensure you have Alembic installed:

```bash
pip install alembic
```

### 2. Run the Migration

```bash
# Initialize Alembic (if not already done)
python setup_migrations.py init

# Check current status
python setup_migrations.py status

# Run the migration
python setup_migrations.py run

# Set up initial embedding models
python setup_migrations.py setup-models
```

### 3. Verify Migration

```python
# Test the migration
python -c "
import asyncio
from src.rag_system.db.pgvector import db

async def check():
    await db.initialize()
    dimensions = await db.get_supported_dimensions()
    print(f'Supported dimensions: {dimensions}')
    await db.close()

asyncio.run(check())
"
```

## 📊 What Changes

### Database Schema

**Before:**
```sql
-- Fixed dimension
embedding vector(1536)
```

**After:**
```sql
-- Multiple columns for different dimensions
embedding_384 vector(384)    -- For all-MiniLM-L6-v2
embedding_768 vector(768)    -- For all-mpnet-base-v2
embedding_1536 vector(1536)  -- For text-embedding-ada-002
embedding_3072 vector(3072)  -- For text-embedding-3-large
```

### New Tables

1. **embedding_models** - Tracks available models and their dimensions
2. Dynamic columns in **chunk_embeddings** based on dimensions needed

## 🔧 Using Different Models

### Ingesting with Specific Models

```python
from src.rag_system.services.ingestion import ingestion_service

# Using OpenAI's large model (3072 dimensions)
doc_id = await ingestion_service.ingest_document(
    title="My Document",
    content="Content here...",
    embedding_model="text-embedding-3-large"
)

# Using a fast local model (384 dimensions)
doc_id = await ingestion_service.ingest_document(
    title="Another Document",
    content="More content...",
    embedding_model="all-MiniLM-L6-v2"
)
```

### Searching with Different Models

```python
from src.rag_system.services.retriever import retriever

# Search using the same model as ingestion
results = await retriever.search(
    query="What is RAG?",
    model_name="text-embedding-3-large"
)

# Search across multiple models
results = await retriever.search_multi_model(
    query="What is RAG?",
    model_names=["text-embedding-ada-002", "all-MiniLM-L6-v2"],
    aggregation="union"  # or "intersection"
)
```

### Migrating Documents to New Models

```python
from src.rag_system.services.ingestion import ingestion_service

# Update a single document
await ingestion_service.update_document_embeddings(
    document_id=doc_id,
    model_name="text-embedding-3-large"
)

# Migrate all documents from one model to another
await ingestion_service.migrate_to_new_model(
    source_model="text-embedding-ada-002",
    target_model="text-embedding-3-large"
)
```

## 📈 Performance Considerations

### Index Management

Each dimension gets its own HNSW index for optimal performance:
```sql
idx_embeddings_hnsw_384
idx_embeddings_hnsw_768
idx_embeddings_hnsw_1536
idx_embeddings_hnsw_3072
```

### Storage Impact

- Each additional dimension adds a new column
- Only populated columns use storage
- Null columns have minimal overhead

### Query Performance

- Searches only scan the relevant dimension column
- HNSW indexes maintain fast search times
- Model-specific queries avoid scanning unnecessary data

## 🔄 Rollback Plan

If you need to rollback (not recommended):

```bash
# Restore from backup
psql -h localhost -U user -d ragdb < ragdb_backup_TIMESTAMP.sql
```

## 📝 API Changes

### New Endpoints

```python
# Get available models for search
GET /api/v1/models/search

# Get document embedding status
GET /api/v1/documents/{document_id}/embeddings

# Update document embeddings
PUT /api/v1/documents/{document_id}/embeddings?model=text-embedding-3-large
```

### Updated Request Schema

```json
{
  "query": "Your question here",
  "model": "text-embedding-3-large",  // Optional, defaults to current model
  "top_k": 5
}
```

## 🎯 Best Practices

1. **Model Selection**
   - Use smaller models (384d) for fast, general search
   - Use larger models (3072d) for high-precision tasks
   - Consider cost vs. performance tradeoffs

2. **Migration Strategy**
   - Start with a small subset of documents
   - Monitor performance and quality
   - Gradually migrate remaining documents

3. **Multi-Model Search**
   - Use for A/B testing different models
   - Combine results for better coverage
   - Monitor which models perform best

## 🐛 Troubleshooting

### Common Issues

1. **"Model exists with different dimension"**
   - A model is already registered with a different dimension
   - Check `embedding_models` table
   - Use consistent model configurations

2. **"Embedding dimension mismatch"**
   - The embedding size doesn't match the model's expected dimension
   - Verify model configuration
   - Check embedding generation code

3. **"Column does not exist"**
   - The vector column for that dimension hasn't been created
   - Run `setup-models` command
   - Or manually add the column

### Debug Commands

```sql
-- Check registered models
SELECT * FROM embedding_models;

-- Check available dimensions
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'chunk_embeddings' 
AND column_name LIKE 'embedding_%';

-- Count embeddings by model
SELECT em.model_name, em.dimension, COUNT(ce.id) as count
FROM embedding_models em
LEFT JOIN chunk_embeddings ce ON ce.model_id = em.id
GROUP BY em.model_name, em.dimension;
```

## 📚 Additional Resources

- [PgVector Documentation](https://github.com/pgvector/pgvector)
- [Embedding Model Comparison](https://huggingface.co/spaces/mteb/leaderboard)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs in `logs/` directory
3. Check database constraints and indexes
4. Open an issue with error details and migration logs