# src/rag_system/services/embeddings.py
from typing import List, Dict, Any, Optional
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from abc import ABC, abstractmethod

from ..core.config import settings

class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass

class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.model_name = model_name
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [e.embedding for e in response.data]
    
    def get_dimension(self) -> int:
        return self.dimensions.get(self.model_name, 1536)

class LocalEmbedding(EmbeddingModel):
    """Local sentence transformer model"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        # Convert to numpy array and then to list
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class EmbeddingService:
    """Main embedding service with model management"""
    
    def __init__(self):
        self.models: Dict[str, EmbeddingModel] = {}
        self.current_model_name = settings.default_embedding_model
        self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize the default embedding model"""
        if self.current_model_name.startswith("text-embedding"):
            self.models[self.current_model_name] = OpenAIEmbedding(self.current_model_name)
        else:
            self.models[self.current_model_name] = LocalEmbedding(self.current_model_name)
    
    def get_current_model(self) -> EmbeddingModel:
        """Get the current embedding model"""
        if self.current_model_name not in self.models:
            self._initialize_default_model()
        return self.models[self.current_model_name]
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        model = self.get_current_model()
        embeddings = await model.embed_texts([text])
        return embeddings[0]
    
    async def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embed multiple texts with batching"""
        model = self.get_current_model()
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await model.embed_texts(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def switch_model(self, model_name: str):
        """Switch to a different embedding model"""
        self.current_model_name = model_name
        if model_name not in self.models:
            if model_name.startswith("text-embedding"):
                self.models[model_name] = OpenAIEmbedding(model_name)
            else:
                self.models[model_name] = LocalEmbedding(model_name)

# Global instance
embedding_service = EmbeddingService()