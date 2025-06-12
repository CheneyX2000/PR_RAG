# docs/README_circuit_breaker.md
# Circuit Breaker Implementation Guide

## Overview

Circuit breakers have been integrated into the RAG system to prevent cascading failures and improve system resilience. They monitor service calls and temporarily block requests to failing services, giving them time to recover.

## What We've Implemented

### 1. **Core Circuit Breaker Module** (`src/rag_system/utils/circuit_breaker.py`)
- Complete circuit breaker pattern implementation
- Three states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)
- Configurable thresholds and timeouts
- Built-in statistics and monitoring

### 2. **Service Integrations**
- **Embeddings Service**: Protected OpenAI API calls
- **Generator Service**: Protected LLM API calls (OpenAI, Anthropic)
- **Database Service**: Protected PgVector operations
- **Redis Service**: Protected cache operations

### 3. **API Endpoints**
- `GET /api/v1/circuit-breakers` - View all circuit breaker statuses
- `GET /api/v1/circuit-breakers/{name}` - Get specific breaker details
- `POST /api/v1/circuit-breakers/{name}/reset` - Manually reset a breaker
- `GET /api/v1/health/detailed` - Health check including circuit breakers

### 4. **Pre-configured Circuit Breakers**
```python
CircuitBreakers.openai      # For OpenAI API calls
CircuitBreakers.database    # For database operations
CircuitBreakers.redis       # For cache operations
CircuitBreakers.external_api # Generic external API
```

## How Circuit Breakers Work

### State Transitions
```
CLOSED → (failures exceed threshold) → OPEN
OPEN → (recovery timeout expires) → HALF_OPEN
HALF_OPEN → (success threshold met) → CLOSED
HALF_OPEN → (any failure) → OPEN
```

### Configuration Options
```python
CircuitBreakerConfig(
    failure_threshold=5,      # Failures before opening
    recovery_timeout=30,      # Seconds before trying again
    success_threshold=2,      # Successes needed to close
    timeout=10.0,            # Call timeout in seconds
    expected_exceptions=...,  # Exceptions that trigger breaker
    exclude_exceptions=...    # Exceptions to ignore
)
```

## Usage Examples

### Basic Usage
```python
from src.rag_system.utils.circuit_breaker import CircuitBreakers

# Using pre-configured breaker
try:
    result = await CircuitBreakers.openai.call_async(
        openai_api_call,
        prompt="Hello"
    )
except CircuitBreakerError:
    # Service is unavailable, use fallback
    result = "Service temporarily unavailable"
```

### Creating Custom Circuit Breakers
```python
from src.rag_system.utils.circuit_breaker import create_circuit_breaker

# Create a custom breaker
my_breaker = create_circuit_breaker(
    name="my_service",
    failure_threshold=3,
    recovery_timeout=60,
    timeout=15.0
)

# Use it
result = await my_breaker.call_async(my_service_call, arg1, arg2)
```

### Using as Decorator
```python
@CircuitBreakers.openai.protect
async def call_openai_api(prompt: str):
    # Your OpenAI API call here
    return response
```

### Monitoring Circuit Breakers
```python
# Get all circuit breaker stats
stats = CircuitBreaker.get_all_stats()

# Get specific breaker status
openai_stats = CircuitBreakers.openai.get_stats()
print(f"State: {openai_stats['state']}")
print(f"Failure rate: {openai_stats['failure_rate']}")
```

## API Usage

### Check Circuit Breaker Status
```bash
# Get all circuit breakers status
curl http://localhost:8000/api/v1/circuit-breakers

# Response:
{
  "circuit_breakers": {
    "openai": {
      "state": "closed",
      "total_calls": 150,
      "successful_calls": 147,
      "failed_calls": 3,
      "rejected_calls": 0,
      "failure_rate": 0.02
    },
    ...
  },
  "summary": {
    "total_breakers": 4,
    "open_breakers": [],
    "degraded_services": []
  }
}
```

### Reset a Circuit Breaker
```bash
curl -X POST http://localhost:8000/api/v1/circuit-breakers/openai/reset
```

### Detailed Health Check
```bash
curl http://localhost:8000/api/v1/health/detailed
```

## Testing Circuit Breakers

### Run Unit Tests
```bash
pytest tests/test_circuit_breaker.py -v
```

### Run Demo Script
```bash
python demo_circuit_breaker.py
```

## Best Practices

1. **Configure Appropriately**
   - Set failure thresholds based on service SLAs
   - Use shorter recovery timeouts for critical services
   - Adjust timeouts based on expected response times

2. **Handle CircuitBreakerError**
   - Always have fallback behavior
   - Log circuit breaker events
   - Monitor circuit breaker metrics

3. **Use Service-Specific Breakers**
   - Don't share breakers between unrelated services
   - Configure each breaker for its specific service characteristics

4. **Monitor and Alert**
   - Set up alerts for circuit breakers opening
   - Track failure rates and patterns
   - Use metrics to tune configuration

## Troubleshooting

### Circuit Breaker Keeps Opening
- Check service health and logs
- Verify network connectivity
- Review failure threshold settings
- Check for timeout issues

### Circuit Breaker Won't Close
- Ensure recovery timeout has passed
- Verify success threshold is achievable
- Check if service is actually recovered
- Use manual reset if necessary

### Performance Impact
- Circuit breakers add minimal overhead
- Failed calls return immediately when open
- Statistics collection is lightweight

## Next Steps

1. **Set up monitoring dashboards** for circuit breaker metrics
2. **Configure alerts** for circuit breaker state changes
3. **Tune thresholds** based on production patterns
4. **Add circuit breakers** to any new external service integrations

## Summary

Circuit breakers are now protecting all critical service dependencies in the RAG system:
- ✅ OpenAI/LLM API calls
- ✅ Database operations
- ✅ Cache operations
- ✅ External API calls

The system will now gracefully degrade instead of cascading failures when services are unavailable.