# PR_RAG
A production-ready RAG (Retrieval-Augmented Generation) system built with Python, FastAPI, PgVector, and model hot-switching capabilities.
# RAG System - Modern Retrieval-Augmented Generation

A production-ready RAG (Retrieval-Augmented Generation) system built with Python, FastAPI, PgVector, and model hot-switching capabilities.

## 🚀 Features

- **Vector Database Integration**: Uses PgVector for efficient similarity search with HNSW indexing
- **Model Hot-Switching**: Dynamically switch between embedding and LLM models without downtime
- **Async Architecture**: Built on FastAPI with full async/await support for high performance
- **Streaming Responses**: Real-time streaming of LLM responses
- **Hybrid Search**: Combines semantic and keyword search (coming soon)
- **Production Ready**: Includes monitoring, error handling, and Docker deployment

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 15+ with PgVector extension
- Redis (optional, for caching)
- OpenAI API key (or other LLM provider keys)

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/CheneyX2000/another_RAG.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# At minimum, set:
# - DATABASE_URL
# - OPENAI_API_KEY
```

### 3. Start Infrastructure

```bash
# Start PostgreSQL and Redis
docker-compose -f docker-compose.dev.yml up -d

# Verify services are running
docker-compose -f docker-compose.dev.yml ps
```

### 4. Initialize and Test

```bash
# Run the quickstart script
python quickstart.py

# This will:
# - Initialize the database
# - Create necessary tables and indexes
# - Ingest sample documents
# - Test retrieval functionality
```

### 5. Start the API Server

```bash
# Development mode with auto-reload
uvicorn src.rag_system.main:app --reload

# Production mode
uvicorn src.rag_system.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Visit http://localhost:8000/docs for interactive API documentation.

## 📚 API Usage Examples

### Ingest a Document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Introduction to LLMs",
    "content": "Large Language Models are...",
    "metadata": {"category": "AI"}
  }'
```

### Search Documents

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are LLMs?",
    "top_k": 5
  }'
```

### Query with Generation

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain how RAG systems work",
    "model": "gpt-4o-mini"
  }'
```

### Stream Response

```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "What is PgVector?"
  }'
```

## 🏗️ Project Structure

```
rag-project/
├── src/rag_system/
│   ├── api/          # FastAPI routes and schemas
│   ├── core/         # Configuration and dependencies
│   ├── db/           # Database models and PgVector
│   ├── services/     # Business logic (retrieval, generation, etc.)
│   └── utils/        # Utilities and helpers
├── tests/            # Test suite
├── docker/           # Docker configurations
└── docs/             # Additional documentation
```

## 🔧 Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI models |
| `DEFAULT_EMBEDDING_MODEL` | Default embedding model | `text-embedding-ada-002` |
| `DEFAULT_LLM_MODEL` | Default LLM for generation | `gpt-4o-mini` |
| `MAX_CHUNK_SIZE` | Maximum chunk size in tokens | `500` |
| `RETRIEVAL_TOP_K` | Default number of results | `5` |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rag_system

# Run specific test file
pytest tests/test_retrieval.py -v
```

## Docker Deployment

### Development

```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Production

```bash
docker-compose up -d
```

## Monitoring

The system includes built-in monitoring with Prometheus metrics:

- Query count and duration
- Model loading statistics  
- Error rates
- Cache hit rates

Access metrics at: http://localhost:8000/metrics

## Model Hot-Switching

Switch embedding models without downtime:

```python
# Via API
PUT /api/v1/documents/{document_id}/embeddings
{
  "model_name": "text-embedding-3-large"
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache2.0 License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/) for RAG orchestration
- Uses [PgVector](https://github.com/pgvector/pgvector) for vector similarity search
- Powered by [FastAPI](https://fastapi.tiangolo.com/) for the API layer
- LLM integration via [LiteLLM](https://github.com/BerriAI/litellm)
