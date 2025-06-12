# src/rag_system/services/cache.py
import json

class RAGCache:
    def __init__(self, redis_client, semantic_threshold=0.95):
        self.redis = redis_client
        self.semantic_threshold = semantic_threshold
        self.embedding_cache = {}
        
    async def get_or_compute(self, key: str, compute_func, ttl=3600):
        """Cache with semantic similarity matching"""
        # Try exact match first
        cached = await self.redis.get(f"rag:{key}")
        if cached:
            return json.loads(cached)
        
        # Try semantic match for queries
        if key.startswith("query:"):
            similar_result = await self._find_semantic_match(key)
            if similar_result:
                return similar_result
        
        # Compute and cache
        result = await compute_func()
        await self.redis.setex(
            f"rag:{key}",
            ttl,
            json.dumps(result)
        )
        return result