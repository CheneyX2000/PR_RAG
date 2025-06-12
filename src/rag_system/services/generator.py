# src/rag_system/services/generator.py
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio
from litellm import acompletion, completion_cost
import openai

from ..services.retriever import RetrievedChunk
from ..core.config import settings
from ..utils.monitoring import logger
from ..utils.exceptions import GenerationError
from ..utils.circuit_breaker import CircuitBreakers, CircuitBreakerError, create_circuit_breaker


@dataclass
class GenerationResponse:
    """Response from the generation service"""
    text: str
    model_name: str
    token_count: int
    cost: Optional[float] = None
    metadata: Dict[str, Any] = None


class GeneratorService:
    """Service for generating responses using LLMs with retrieved context"""
    
    def __init__(self):
        self.default_system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Always cite the sources when using information from the context.
        If the context doesn't contain relevant information, say so clearly."""
        
        # Model configuration
        self.model_configs = {
            "gpt-4o-mini": {
                "max_tokens": 2000,
                "temperature": 0.7,
                "provider": "openai"
            },
            "gpt-4o": {
                "max_tokens": 4000,
                "temperature": 0.7,
                "provider": "openai"
            },
            "claude-3-haiku-20240307": {
                "max_tokens": 2000,
                "temperature": 0.7,
                "provider": "anthropic"
            },
            "claude-3-sonnet-20240229": {
                "max_tokens": 4000,
                "temperature": 0.7,
                "provider": "anthropic"
            }
        }
        
        # Create circuit breakers for different providers
        self.circuit_breakers = {
            "openai": CircuitBreakers.openai,
            "anthropic": create_circuit_breaker(
                name="anthropic",
                failure_threshold=3,
                recovery_timeout=60,
                timeout=30.0
            ),
            "default": CircuitBreakers.external_api
        }
    
    def _get_circuit_breaker(self, model_name: str):
        """Get the appropriate circuit breaker for a model"""
        config = self.model_configs.get(model_name, {})
        provider = config.get("provider", "default")
        return self.circuit_breakers.get(provider, self.circuit_breakers["default"])
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into context string"""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.document_title}]\n{chunk.content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_messages(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build messages for the LLM"""
        system_prompt = system_prompt or self.default_system_prompt
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": f"""Context information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain relevant information, please state that clearly."""
            }
        ]
        
        return messages
    
    async def _call_llm(self, **kwargs):
        """Make the actual LLM API call"""
        return await acompletion(**kwargs)
    
    async def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> GenerationResponse:
        """
        Generate a response using the LLM with retrieved context
        
        Args:
            query: User query
            context_chunks: Retrieved document chunks
            model_name: LLM model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Custom system prompt
            
        Returns:
            Generated response with metadata
        """
        model_name = model_name or settings.default_llm_model
        circuit_breaker = self._get_circuit_breaker(model_name)
        
        try:
            # Format context
            context = self._format_context(context_chunks)
            
            # Build messages
            messages = self._build_messages(query, context, system_prompt)
            
            # Get model config
            model_config = self.model_configs.get(model_name, {})
            
            # Prepare completion parameters
            completion_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature or model_config.get("temperature", 0.7),
                "max_tokens": max_tokens or model_config.get("max_tokens", 2000),
            }
            
            # Log generation request
            logger.info(
                "generation_start",
                model=model_name,
                query_length=len(query),
                context_chunks=len(context_chunks),
                temperature=completion_params["temperature"]
            )
            
            # Call LLM with circuit breaker protection
            response = await circuit_breaker.call_async(
                self._call_llm,
                **completion_params
            )
            
            # Extract response
            generated_text = response.choices[0].message.content
            token_count = response.usage.total_tokens if response.usage else 0
            
            # Calculate cost if available
            try:
                cost = completion_cost(completion_response=response)
            except:
                cost = None
            
            logger.info(
                "generation_complete",
                model=model_name,
                tokens_used=token_count,
                cost=cost
            )
            
            return GenerationResponse(
                text=generated_text,
                model_name=model_name,
                token_count=token_count,
                cost=cost,
                metadata={
                    "context_chunks_used": len(context_chunks),
                    "temperature": completion_params["temperature"],
                    "max_tokens": completion_params["max_tokens"]
                }
            )
            
        except CircuitBreakerError as e:
            logger.error(
                "generation_circuit_breaker_open",
                model=model_name,
                error=str(e)
            )
            raise GenerationError(
                f"LLM service for {model_name} is temporarily unavailable. Please try again later."
            )
        except Exception as e:
            logger.error(
                "generation_error",
                model=model_name,
                error=str(e)
            )
            raise GenerationError(f"Failed to generate response: {str(e)}")
    
    async def stream_generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream generate a response token by token
        
        Args:
            query: User query
            context_chunks: Retrieved document chunks
            model_name: LLM model to use
            temperature: Generation temperature
            system_prompt: Custom system prompt
            
        Yields:
            Generated text tokens
        """
        model_name = model_name or settings.default_llm_model
        circuit_breaker = self._get_circuit_breaker(model_name)
        
        try:
            # Format context
            context = self._format_context(context_chunks)
            
            # Build messages
            messages = self._build_messages(query, context, system_prompt)
            
            # Get model config
            model_config = self.model_configs.get(model_name, {})
            
            # Prepare completion parameters
            completion_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature or model_config.get("temperature", 0.7),
                "max_tokens": model_config.get("max_tokens", 2000),
                "stream": True
            }
            
            # Create async wrapper for streaming with circuit breaker
            async def _stream_with_circuit_breaker():
                return await circuit_breaker.call_async(
                    self._call_llm,
                    **completion_params
                )
            
            # Get the stream
            stream = await _stream_with_circuit_breaker()
            
            # Stream tokens
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except CircuitBreakerError as e:
            logger.error(
                "stream_generation_circuit_breaker_open",
                model=model_name,
                error=str(e)
            )
            raise GenerationError(
                f"LLM service for {model_name} is temporarily unavailable. Please try again later."
            )
        except Exception as e:
            logger.error(
                "stream_generation_error",
                model=model_name,
                error=str(e)
            )
            raise GenerationError(f"Failed to stream response: {str(e)}")
    
    def validate_model(self, model_name: str) -> bool:
        """Check if a model is supported"""
        return model_name in self.model_configs
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.model_configs.keys())
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for all LLM providers"""
        return {
            provider: breaker.get_stats()
            for provider, breaker in self.circuit_breakers.items()
        }


# Global instance
generator_service = GeneratorService()