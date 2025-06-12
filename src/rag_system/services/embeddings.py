# src/rag_system/services/embeddings.py
from typing import List, Dict, Any, Optional, Tuple
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from abc import ABC, abstractmethod

from ..core.config import settings
from ..utils.exceptions import EmbeddingError, ValidationError
from ..utils.monitoring import logger, embedding_generation_duration
from ..utils.circuit_breaker import CircuitBreakers, CircuitBreakerError

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass

class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model with circuit breaker protection"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
    
    async def _call_openai_api(self, texts: List[str]):
        """Make the actual API call to OpenAI"""
        return await self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with circuit breaker protection"""
        if not texts:
            return []
        
        try:
            with embedding_generation_duration.labels(model=self.model_name).time():
                # Use circuit breaker for OpenAI API calls
                response = await CircuitBreakers.openai.call_async(
                    self._call_openai_api,
                    texts
                )
            return [e.embedding for e in response.data]
            
        except CircuitBreakerError as e:
            logger.error(f"OpenAI circuit breaker open: {e}")
            raise EmbeddingError(
                f"OpenAI service is temporarily unavailable. Please try again later."
            )
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
    
    def get_dimension(self) -> int:
        return self.dimensions.get(self.model_name, 1536)
    
    def get_model_name(self) -> str:
        return self.model_name

class LocalEmbedding(EmbeddingModel):
    """Local sentence transformer model"""
    
    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
            # Get actual dimension from model
            self._dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load local model {model_name}: {e}")
            raise EmbeddingError(f"Failed to load embedding model: {str(e)}")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        try:
            with embedding_generation_duration.labels(model=self.model_name).time():
                # Run in thread pool to avoid blocking
                import asyncio
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    lambda: self.model.encode(texts)
                )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Local embedding error: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
    
    def get_dimension(self) -> int:
        return self._dimension
    
    def get_model_name(self) -> str:
        return self.model_name

class EmbeddingService:
    """Main embedding service with model management and circuit breaker support"""
    
    def __init__(self):
        self.models: Dict[str, EmbeddingModel] = {}
        self.current_model_name = settings.default_embedding_model
        self._model_dimensions: Dict[str, int] = {}
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize the default embedding model"""
        try:
            if self.current_model_name.startswith("text-embedding"):
                model = OpenAIEmbedding(self.current_model_name)
            else:
                model = LocalEmbedding(self.current_model_name)
            
            self.models[self.current_model_name] = model
            self._model_dimensions[self.current_model_name] = model.get_dimension()
            
            logger.info(
                f"Initialized embedding model: {self.current_model_name} "
                f"with dimension {model.get_dimension()}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize default embedding model: {e}")
            raise
    
    def get_current_model(self) -> EmbeddingModel:
        """Get the current embedding model"""
        if self.current_model_name not in self.models:
            self._initialize_default_model()
        return self.models[self.current_model_name]
    
    def get_model_dimension(self, model_name: Optional[str] = None) -> int:
        """Get the dimension for a model"""
        model_name = model_name or self.current_model_name
        
        if model_name in self._model_dimensions:
            return self._model_dimensions[model_name]
        
        # Try to load the model to get dimension
        if model_name not in self.models:
            if model_name.startswith("text-embedding"):
                model = OpenAIEmbedding(model_name)
            else:
                model = LocalEmbedding(model_name)
            
            self.models[model_name] = model
            self._model_dimensions[model_name] = model.get_dimension()
        
        return self._model_dimensions[model_name]
    
    async def embed_text(self, text: str, model_name: Optional[str] = None) -> Tuple[List[float], int]:
        """
        Embed a single text and return embedding with dimension
        
        Returns:
            Tuple of (embedding, dimension)
        """
        model_name = model_name or self.current_model_name
        model = self._get_or_create_model(model_name)
        
        embeddings = await model.embed_texts([text])
        dimension = model.get_dimension()
        
        return embeddings[0], dimension
    
    async def embed_texts(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        model_name: Optional[str] = None
    ) -> Tuple[List[List[float]], int]:
        """
        Embed multiple texts with batching
        
        Returns:
            Tuple of (embeddings, dimension)
        """
        if not texts:
            return [], 0
        
        model_name = model_name or self.current_model_name
        model = self._get_or_create_model(model_name)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await model.embed_texts(batch)
            all_embeddings.extend(embeddings)
        
        dimension = model.get_dimension()
        
        # Validate all embeddings have correct dimension
        for i, emb in enumerate(all_embeddings):
            if len(emb) != dimension:
                raise ValidationError(
                    f"Embedding {i} has dimension {len(emb)}, "
                    f"expected {dimension} for model {model_name}"
                )
        
        return all_embeddings, dimension
    
    def _get_or_create_model(self, model_name: str) -> EmbeddingModel:
        """Get existing model or create new one"""
        if model_name not in self.models:
            if model_name.startswith("text-embedding"):
                model = OpenAIEmbedding(model_name)
            else:
                model = LocalEmbedding(model_name)
            
            self.models[model_name] = model
            self._model_dimensions[model_name] = model.get_dimension()
            
            logger.info(
                f"Loaded embedding model: {model_name} "
                f"with dimension {model.get_dimension()}"
            )
        
        return self.models[model_name]
    
    def switch_model(self, model_name: str):
        """Switch to a different embedding model"""
        old_model = self.current_model_name
        old_dimension = self.get_model_dimension(old_model)
        new_dimension = self.get_model_dimension(model_name)
        
        self.current_model_name = model_name
        
        logger.info(
            f"Switched embedding model from {old_model} (dim={old_dimension}) "
            f"to {model_name} (dim={new_dimension})"
        )
        
        if old_dimension != new_dimension:
            logger.warning(
                f"Dimension change detected: {old_dimension} -> {new_dimension}. "
                "Existing embeddings will need migration."
            )
    
    def get_available_models(self) -> Dict[str, int]:
        """Get all available models and their dimensions"""
        available = {}
        
        # OpenAI models
        openai_models = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        available.update(openai_models)
        
        # Common local models
        local_models = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-MiniLM-L12-v2": 384,
        }
        available.update(local_models)
        
        return available
    
    async def validate_embedding(self, embedding: List[float], model_name: str) -> bool:
        """Validate that an embedding matches expected dimension for a model"""
        expected_dim = self.get_model_dimension(model_name)
        actual_dim = len(embedding)
        
        if actual_dim != expected_dim:
            logger.error(
                f"Embedding dimension mismatch for model {model_name}: "
                f"expected {expected_dim}, got {actual_dim}"
            )
            return False
        
        return True
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for embedding services"""
        return {
            "openai": CircuitBreakers.openai.get_stats(),
            "current_model": self.current_model_name,
            "is_openai_model": self.current_model_name.startswith("text-embedding")
        }

# Global instance
embedding_service = EmbeddingService()