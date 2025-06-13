# RAG System - Modern Retrieval-Augmented Generation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready RAG (Retrieval-Augmented Generation) system built with Python, FastAPI, PgVector, and model hot-switching capabilities.

## 🔗 Quick Links

| Resource | Description |
|----------|-------------|
| [📚 API Docs](http://localhost:8000/docs) | Interactive API documentation |
| [🛡️ Circuit Breakers](docs/README_circuit_breaker.md) | Resilience patterns guide |
| [🎯 Reranking](docs/README_reranking.md) | Search quality improvement |
| [🔄 Migrations](docs/README_migrations.md) | Database migration guide |

## 🚀 Features

- **Vector Database Integration**: Uses PgVector for efficient similarity search with HNSW indexing
- **Model Hot-Switching**: Dynamically switch between embedding and LLM models without downtime
- **Cross-Encoder Reranking**: Neural reranking for improved search quality
- **Circuit Breakers**: Built-in resilience for all external service calls
- **Async Architecture**: Built on FastAPI with full async/await support for high performance
- **Streaming Responses**: Real-time streaming of LLM responses
- **Production Ready**: Includes monitoring, error handling, and Docker deployment

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL 15+ with PgVector extension
- Redis (optional, for caching)
- OpenAI API key (or other LLM provider keys)

## 🛠️ Quick Start

### One-Click Installation

#### Linux/macOS
```bash
chmod +x install.sh
./install.sh
```

#### Windows
```cmd
install.bat
```

These scripts will automatically:
- Check all prerequisites
- Create virtual environment
- Install dependencies
- Start infrastructure services
- Initialize the database

### Manual Installation

#### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/CheneyX2000/another_RAG.git
cd another_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

#### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# At minimum, set:
# - DATABASE_URL
# - OPENAI_API_KEY
```

#### 3. Start Infrastructure

```bash
# Start PostgreSQL and Redis
docker-compose -f docker-compose.dev.yml up -d

# Verify services are running
docker-compose -f docker-compose.dev.yml ps
```

#### 4. Initialize and Test

```bash
# Run the quickstart script
python quickstart.py

# This will:
# - Initialize the database
# - Create necessary tables and indexes
# - Ingest sample documents
# - Test retrieval functionality
```

#### 5. Start the API Server

```bash
# Development mode with auto-reload
uvicorn src.rag_system.main:app --reload

# Production mode
uvicorn src.rag_system.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Visit http://localhost:8000/docs for interactive API documentation.

## 🏛️ Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Client    │────▶│   FastAPI +     │────▶│   Services   │
│Application  │     │   Middleware    │     │    Layer     │
└─────────────┘     └─────────────────┘     └──────┬───────┘
                                                    │
                    ┌───────────────────────────────┴─────────────────┐
                    │                                                 │
                    ▼                ▼                ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐
            │  Embedding   │ │  Retriever   │ │  Generator   │ │  Cache  │
            │   Service    │ │   Service    │ │   Service    │ │ Service │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └────┬────┘
                   │                │                 │               │
                   ▼                ▼                 ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐
            │   OpenAI/    │ │  PostgreSQL  │ │    LiteLLM   │ │  Redis  │
            │Local Models  │ │  + PgVector  │ │(Multi-Model) │ │         │
            └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘
```

### Key Components

- **API Layer**: FastAPI with streaming support and comprehensive middleware
- **Circuit Breakers**: Protect all external service calls with automatic recovery
- **Services**: Modular business logic with clean separation of concerns
- **Vector Store**: PgVector with HNSW indexing for fast similarity search
- **Model Management**: Hot-swappable embeddings and LLM models
- **Caching**: Redis integration for performance optimization

## 📚 API Usage Examples

### Basic Document Ingestion

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Introduction to LLMs",
    "content": "Large Language Models are...",
    "metadata": {"category": "AI"}
  }'
```

### Search with Reranking

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are LLMs?",
    "top_k": 20,
    "rerank": true,
    "rerank_model": "ms-marco-MiniLM-L-6-v2",
    "rerank_top_k": 5
  }'
```

### Multi-Model Search

```bash
curl -X POST "http://localhost:8000/api/v1/search/multi-model" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explain vector databases",
    "model_names": ["text-embedding-ada-002", "all-MiniLM-L6-v2"],
    "aggregation": "union"
  }'
```

### Query with Generation (Streaming)

```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "Explain how RAG systems work",
    "model": "gpt-4o-mini",
    "temperature": 0.7
  }'
```

## 🎯 Advanced Features

### Cross-Encoder Reranking

Improve search quality by reranking initial results with a cross-encoder model:

```python
# Enable reranking for better relevance
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "Your question here",
        "top_k": 50,  # Get more initial results
        "rerank": True,
        "rerank_top_k": 10  # Keep top 10 after reranking
    }
)
```

Available reranking models:
- `ms-marco-TinyBERT-L-2-v2` (fastest)
- `ms-marco-MiniLM-L-6-v2` (balanced)
- `ms-marco-MiniLM-L-12-v2` (highest quality)

### Circuit Breakers

Monitor and manage service resilience:

```bash
# Check all circuit breakers status
curl http://localhost:8000/api/v1/circuit-breakers

# Reset a specific circuit breaker
curl -X POST http://localhost:8000/api/v1/circuit-breakers/openai/reset
```

### Model Hot-Switching

Switch models without restarting:

```bash
# Update embeddings for a document
curl -X PUT "http://localhost:8000/api/v1/documents/{document_id}/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "text-embedding-3-large"
  }'
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
| `CHUNK_OVERLAP` | Overlap between chunks | `50` |
| `RETRIEVAL_TOP_K` | Default number of results | `10` |
| `REDIS_URL` | Redis connection string | Optional |
| `API_KEY_REQUIRED` | Enable API key authentication | `false` |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rag_system --cov-report=html

# Run specific test categories
pytest tests/unit -v
pytest tests/integration -v
pytest tests/e2e -v
```

## 🚀 Production Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# Scale API servers
docker-compose up -d --scale api=3

# View logs
docker-compose logs -f api
```

### Environment Variables for Production

```bash
# Security
API_KEY_REQUIRED=true
ALLOWED_HOSTS=yourdomain.com
CORS_ORIGINS=https://yourdomain.com

# Performance
MAX_WORKERS=4
CONNECTION_POOL_SIZE=20
CACHE_TTL=3600

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### Health Monitoring

- **Health Check**: `GET /api/v1/health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Circuit Breakers**: `GET /api/v1/circuit-breakers`

## 🐛 Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check PostgreSQL is running
docker-compose -f docker-compose.dev.yml ps

# Verify connection string
psql $DATABASE_URL -c "SELECT 1"
```

**OpenAI API Errors**
```bash
# Check API key
echo $OPENAI_API_KEY

# Monitor circuit breaker
curl http://localhost:8000/api/v1/circuit-breakers/openai
```

**Embedding Dimension Mismatch**
```sql
-- Check registered models
SELECT * FROM embedding_models;

-- Check vector columns
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'chunk_embeddings' AND column_name LIKE 'embedding_%';
```

### Debug Mode

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
uvicorn src.rag_system.main:app --reload --log-level debug
```

## ⚡ Performance Tips

- **Batch Processing**: Ingest documents in batches for better throughput
- **Index Tuning**: Adjust HNSW parameters (`m`, `ef_construction`) for your dataset
- **Connection Pooling**: Set pool size based on concurrent users
- **Caching**: Enable Redis for frequently accessed content
- **Model Selection**: Use smaller models for real-time applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PgVector](https://github.com/pgvector/pgvector) for vector similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the modern API framework
- [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM interface
- [Sentence Transformers](https://www.sbert.net/) for embedding models