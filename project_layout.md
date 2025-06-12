rag-project/
├── pyproject.toml              # Primary configuration
├── quickstart.py  
├── README.md
├── .env.example
├── src/
│   └── rag_system/
|       | __init__.py
│       ├── main.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py       # FastAPI routes
│       │   ├── schemas.py      # Pydantic models
│       │   └── middleware.py   # Rate limiting, auth
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py       # Settings management
│       │   ├── dependencies.py # Dependency injection
│       │   └── models.py       # Domain models
│       ├── services/
│       │   ├── __init__.py
│       │   ├── retriever.py    # Document retrieval
│       │   ├── embeddings.py   # Embedding generation
|       |   ├── ingestions.py   # Ingesting documents into RAG
│       │   ├── generator.py    # LLM integration
│       │   └── cache.py        # Caching layer
│       ├── db/
│       │   ├── __init__.py
│       │   ├── pgvector.py     # PgVector operations
│       │   ├── models.py       # SQLAlchemy models
│       │   └── migrations/     # Alembic migrations
│       └── utils/
│           ├── __init__.py
│           ├── chunking.py     # Text chunking
│           ├── monitoring.py   # Metrics & logging
│           └── excepti ons.py   # Custom exceptions
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    └── architecture.md