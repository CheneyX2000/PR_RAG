# src/rag_system/utils/monitoring.py
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram(
    'rag_query_duration_seconds', 
    'Query duration',
    ['method', 'endpoint', 'status']
)
active_models = Gauge('rag_active_models', 'Number of loaded models')

# Additional metrics
document_ingestion_counter = Counter(
    'rag_documents_ingested_total',
    'Total documents ingested',
    ['document_type']
)
document_ingestion_duration = Histogram(
    'rag_document_ingestion_duration_seconds',
    'Document ingestion duration'
)
embedding_generation_duration = Histogram(
    'rag_embedding_generation_duration_seconds',
    'Embedding generation duration',
    ['model']
)
llm_generation_duration = Histogram(
    'rag_llm_generation_duration_seconds',
    'LLM generation duration',
    ['model']
)
cache_hits = Counter(
    'rag_cache_hits_total',
    'Total cache hits',
    ['cache_type']
)
cache_misses = Counter(
    'rag_cache_misses_total',
    'Total cache misses',
    ['cache_type']
)
error_counter = Counter(
    'rag_errors_total',
    'Total errors',
    ['error_type', 'operation']
)

# Create logger instance
logger = structlog.get_logger()

# Export metrics dict for easy access
metrics = {
    'query_counter': query_counter,
    'query_duration': query_duration,
    'active_models': active_models,
    'document_ingestion_counter': document_ingestion_counter,
    'document_ingestion_duration': document_ingestion_duration,
    'embedding_generation_duration': embedding_generation_duration,
    'llm_generation_duration': llm_generation_duration,
    'cache_hits': cache_hits,
    'cache_misses': cache_misses,
    'error_counter': error_counter
}