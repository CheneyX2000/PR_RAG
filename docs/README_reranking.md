# docs/README_reranking.md
# Reranking Implementation Guide

## Overview

The RAG system now includes a sophisticated reranking capability that significantly improves search quality by using cross-encoder models to re-score initial retrieval results. This two-stage approach balances efficiency with accuracy.

## How Reranking Works

### Two-Stage Retrieval Process

1. **Initial Retrieval (Fast)**
   - Uses vector similarity search with embeddings
   - Retrieves a larger candidate set (e.g., top 15-20 documents)
   - Optimized for speed and scalability

2. **Reranking (Accurate)**
   - Uses cross-encoder models to score query-document pairs
   - Captures fine-grained semantic relationships
   - Returns the most relevant documents

### Cross-Encoder vs Bi-Encoder

- **Bi-Encoder** (used in initial retrieval):
  - Encodes query and documents separately
  - Fast but may miss nuanced relationships
  
- **Cross-Encoder** (used in reranking):
  - Processes query and document together
  - Slower but much more accurate
  - Can understand complex query-document interactions

## Available Reranking Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `ms-marco-TinyBERT-L-2-v2` | Fastest | Good | High-volume, real-time applications |
| `ms-marco-MiniLM-L-6-v2` | Balanced | Very Good | Default choice, good balance |
| `ms-marco-MiniLM-L-12-v2` | Slower | Best | When accuracy is critical |

## API Usage

### Basic Search with Reranking

```python
# Enable reranking in search
response = await client.post("/api/v1/search", json={
    "query": "What is reranking in RAG?",
    "top_k": 5,
    "rerank": True,
    "rerank_model": "ms-marco-MiniLM-L-6-v2"  # Optional, uses default if not specified
})
```

### Configure Reranking Globally

```python
# Enable reranking for all searches
await client.put("/api/v1/reranking/config", json={
    "enabled": True,
    "model": "ms-marco-MiniLM-L-12-v2"
})

# Check current configuration
config = await client.get("/api/v1/reranking/status")
```

### Compare Results

```python
# Compare search with and without reranking
comparison = await client.post("/api/v1/search/compare", json={
    "query": "How do cross-encoders work?",
    "top_k": 5
})

# Shows rank changes and improvements
print(comparison["rank_changes"])
print(comparison["summary"])
```

### Multi-Model Search with Reranking

```python
# Search across multiple embedding models and rerank
results = await client.post("/api/v1/search/multi-model", json={
    "query": "Production RAG best practices",
    "model_names": ["text-embedding-ada-002", "all-MiniLM-L6-v2"],
    "aggregation": "union",
    "rerank": True,
    "rerank_model": "ms-marco-MiniLM-L-6-v2"
})
```

## Python SDK Usage

```python
from src.rag_system.services.retriever import retriever

# Enable reranking globally
retriever.enable_reranking(True, model="ms-marco-MiniLM-L-12-v2")

# Search with reranking
results = await retriever.search(
    query="What makes a good RAG system?",
    top_k=5,
    rerank=True,  # Override global setting
    rerank_model="ms-marco-MiniLM-L-6-v2",  # Use specific model
    rerank_top_k=5  # Number of results after reranking
)

# Access both scores
for result in results:
    print(f"Title: {result.document_title}")
    print(f"Vector Similarity: {result.similarity_score:.4f}")
    print(f"Rerank Score: {result.rerank_score:.4f}")
```

## Performance Considerations

### Latency Impact

| Configuration | Typical Latency | Use Case |
|--------------|-----------------|----------|
| No reranking | 50-100ms | Real-time, high-volume |
| TinyBERT reranking | 100-200ms | Balanced performance |
| MiniLM-L-6 reranking | 150-300ms | Default, good quality |
| MiniLM-L-12 reranking | 300-500ms | Best quality |

### Optimization Tips

1. **Adjust Initial Retrieval Size**
   ```python
   # Retrieve more candidates for reranking
   results = await retriever.search(
       query="...",
       top_k=5,  # Final results
       rerank=True
   )
   # System automatically retrieves 3x top_k for reranking
   ```

2. **Use Caching**
   - Reranked results are cached with query + model as key
   - Cache TTL respects your Redis configuration

3. **Batch Processing**
   - Reranking processes documents in configurable batches
   - Adjust batch size based on available memory

4. **Model Loading**
   - Models are lazy-loaded on first use
   - Once loaded, they remain in memory for fast inference

## Configuration Options

### Environment Variables

```bash
# In .env file
DEFAULT_RERANKING_MODEL=ms-marco-MiniLM-L-6-v2
RERANKING_ENABLED=true
RERANKING_BATCH_SIZE=32
```

### Runtime Configuration

```python
# Get available models
models = retriever.get_reranking_models()

# Configure specific model
config = RerankingConfig(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size=16,  # Smaller batch for limited memory
    max_length=512,  # Maximum input length
    normalize_scores=True  # Normalize scores to [0, 1]
)
```

## Testing and Validation

### Run the Demo

```bash
# Install dependencies
pip install rich httpx

# Run the interactive demo
python demo_reranking.py
```

### Run Tests

```bash
# Run reranking tests
pytest tests/test_reranking.py -v

# Run with coverage
pytest tests/test_reranking.py --cov=src.rag_system.services.retriever
```

### Benchmarking

```python
import time

# Benchmark reranking impact
async def benchmark_reranking(query, iterations=10):
    times_without = []
    times_with = []
    
    for _ in range(iterations):
        # Without reranking
        start = time.time()
        await retriever.search(query, rerank=False)
        times_without.append(time.time() - start)
        
        # With reranking
        start = time.time()
        await retriever.search(query, rerank=True)
        times_with.append(time.time() - start)
    
    print(f"Average without reranking: {sum(times_without)/len(times_without):.3f}s")
    print(f"Average with reranking: {sum(times_with)/len(times_with):.3f}s")
```

## Circuit Breaker Protection

Reranking models are protected by circuit breakers:

```python
# Check reranking circuit breaker status
breaker_status = CircuitBreaker.get_instance("reranker_ms-marco-MiniLM-L-6-v2")
print(breaker_status.get_stats())

# Manual reset if needed
breaker_status.reset()
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Use a faster model (TinyBERT)
   - Reduce the number of candidates (lower initial top_k)
   - Check if model is loaded in memory

2. **Out of Memory**
   - Reduce batch size in configuration
   - Use a smaller model
   - Limit the number of concurrent requests

3. **Poor Reranking Quality**
   - Try a more powerful model (MiniLM-L-12)
   - Ensure documents have sufficient content
   - Check that query is well-formed

### Debug Logging

```python
import logging

# Enable debug logging for reranking
logging.getLogger("src.rag_system.services.retriever").setLevel(logging.DEBUG)
```

## Best Practices

1. **Always Enable for User-Facing Search**
   - The quality improvement is significant
   - Users expect the most relevant results

2. **Choose Model Based on Use Case**
   - Real-time: TinyBERT
   - Balanced: MiniLM-L-6 (default)
   - Research/Analytics: MiniLM-L-12

3. **Monitor Performance**
   - Track reranking latency
   - Monitor model memory usage
   - Watch circuit breaker status

4. **A/B Testing**
   - Use the compare endpoint to validate improvements
   - Track user engagement metrics
   - Adjust models based on feedback

## Advanced Usage

### Custom Reranking Models

```python
# Add a custom cross-encoder model
custom_config = RerankingConfig(
    model_name="your-org/custom-cross-encoder",
    batch_size=16,
    max_length=256,
    normalize_scores=True
)

retriever.RERANKING_MODELS["custom"] = custom_config
retriever.enable_reranking(True, model="custom")
```

### Hybrid Scoring

```python
# Combine vector similarity and rerank scores
def hybrid_score(chunk, alpha=0.7):
    """Weighted combination of scores"""
    return (alpha * chunk.rerank_score + 
            (1 - alpha) * chunk.similarity_score)

# Apply to results
results = await retriever.search(query, rerank=True)
for chunk in results:
    chunk.hybrid_score = hybrid_score(chunk)

# Resort by hybrid score
results.sort(key=lambda x: x.hybrid_score, reverse=True)
```

## Conclusion

Reranking is a powerful technique that significantly improves RAG search quality with manageable latency overhead. The implementation is production-ready with:

- ✅ Multiple model options for different use cases
- ✅ Circuit breaker protection for resilience
- ✅ Efficient batching and caching
- ✅ Easy configuration and monitoring
- ✅ Comprehensive testing and validation

Start with the default configuration and adjust based on your specific requirements!