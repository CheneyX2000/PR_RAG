# tests/test_generator.py
"""
Comprehensive tests for the generator service.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from src.rag_system.services.generator import (
    GeneratorService,
    GenerationResponse
)
from src.rag_system.services.retriever import RetrievedChunk
from src.rag_system.utils.exceptions import GenerationError
from src.rag_system.utils.circuit_breaker import CircuitBreakerError


class TestGeneratorService:
    """Test cases for the generator service"""
    
    @pytest.fixture
    def generator_service_instance(self):
        """Create generator service instance"""
        return GeneratorService()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample retrieved chunks"""
        return [
            RetrievedChunk(
                document_id="doc1",
                document_title="Test Document 1",
                chunk_id="chunk1",
                content="This is test content about RAG systems.",
                chunk_index=0,
                similarity_score=0.95
            ),
            RetrievedChunk(
                document_id="doc2",
                document_title="Test Document 2",
                chunk_id="chunk2",
                content="PgVector is used for vector similarity search.",
                chunk_index=0,
                similarity_score=0.92
            )
        ]
    
    def test_format_context(self, generator_service_instance, sample_chunks):
        """Test context formatting from chunks"""
        context = generator_service_instance._format_context(sample_chunks)
        
        assert "[Source 1: Test Document 1]" in context
        assert "This is test content about RAG systems." in context
        assert "[Source 2: Test Document 2]" in context
        assert "PgVector is used for vector similarity search." in context
    
    def test_format_context_empty(self, generator_service_instance):
        """Test context formatting with no chunks"""
        context = generator_service_instance._format_context([])
        assert context == "No relevant context found."
    
    def test_build_messages(self, generator_service_instance, sample_chunks):
        """Test message building for LLM"""
        context = generator_service_instance._format_context(sample_chunks)
        messages = generator_service_instance._build_messages(
            query="What is RAG?",
            context=context
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "helpful AI assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "What is RAG?" in messages[1]["content"]
        assert context in messages[1]["content"]
    
    def test_build_messages_custom_prompt(self, generator_service_instance):
        """Test message building with custom system prompt"""
        custom_prompt = "You are a technical expert."
        messages = generator_service_instance._build_messages(
            query="Test query",
            context="Test context",
            system_prompt=custom_prompt
        )
        
        assert messages[0]["content"] == custom_prompt
    
    def test_get_circuit_breaker(self, generator_service_instance):
        """Test circuit breaker selection"""
        # OpenAI model
        breaker = generator_service_instance._get_circuit_breaker("gpt-4o-mini")
        assert breaker == generator_service_instance.circuit_breakers["openai"]
        
        # Anthropic model
        breaker = generator_service_instance._get_circuit_breaker("claude-3-haiku-20240307")
        assert breaker == generator_service_instance.circuit_breakers["anthropic"]
        
        # Unknown model
        breaker = generator_service_instance._get_circuit_breaker("unknown-model")
        assert breaker == generator_service_instance.circuit_breakers["default"]
    
    @pytest.mark.asyncio
    async def test_generate_success(self, generator_service_instance, sample_chunks):
        """Test successful generation"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated response"))]
        mock_response.usage = Mock(total_tokens=150)
        
        with patch.object(generator_service_instance, '_call_llm', return_value=mock_response):
            with patch('litellm.completion_cost', return_value=0.005):
                response = await generator_service_instance.generate(
                    query="What is RAG?",
                    context_chunks=sample_chunks,
                    model_name="gpt-4o-mini",
                    temperature=0.7,
                    max_tokens=500
                )
                
                assert isinstance(response, GenerationResponse)
                assert response.text == "Generated response"
                assert response.model_name == "gpt-4o-mini"
                assert response.token_count == 150
                assert response.cost == 0.005
                assert response.metadata["context_chunks_used"] == 2
    
    @pytest.mark.asyncio
    async def test_generate_with_default_model(self, generator_service_instance, sample_chunks):
        """Test generation with default model"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Default model response"))]
        mock_response.usage = Mock(total_tokens=100)
        
        with patch.object(generator_service_instance, '_call_llm', return_value=mock_response):
            with patch('src.rag_system.core.config.settings') as mock_settings:
                mock_settings.default_llm_model = "gpt-4o-mini"
                
                response = await generator_service_instance.generate(
                    query="Test query",
                    context_chunks=sample_chunks
                )
                
                assert response.model_name == "gpt-4o-mini"
    
    @pytest.mark.asyncio
    async def test_generate_circuit_breaker_error(self, generator_service_instance, sample_chunks):
        """Test generation when circuit breaker is open"""
        with patch.object(generator_service_instance._get_circuit_breaker("gpt-4o-mini"), 
                         'call_async') as mock_cb:
            mock_cb.side_effect = CircuitBreakerError("Circuit open")
            
            with pytest.raises(GenerationError) as exc_info:
                await generator_service_instance.generate(
                    query="Test query",
                    context_chunks=sample_chunks,
                    model_name="gpt-4o-mini"
                )
            
            assert "temporarily unavailable" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_api_error(self, generator_service_instance, sample_chunks):
        """Test generation with API error"""
        with patch.object(generator_service_instance, '_call_llm') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            with pytest.raises(GenerationError) as exc_info:
                await generator_service_instance.generate(
                    query="Test query",
                    context_chunks=sample_chunks
                )
            
            assert "Failed to generate response" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_cost_calculation_error(self, generator_service_instance, sample_chunks):
        """Test generation when cost calculation fails"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = Mock(total_tokens=100)
        
        with patch.object(generator_service_instance, '_call_llm', return_value=mock_response):
            with patch('litellm.completion_cost', side_effect=Exception("Cost error")):
                response = await generator_service_instance.generate(
                    query="Test query",
                    context_chunks=sample_chunks
                )
                
                assert response.cost is None  # Cost calculation failed gracefully
    
    @pytest.mark.asyncio
    async def test_stream_generate_success(self, generator_service_instance, sample_chunks):
        """Test successful streaming generation"""
        # Create mock stream chunks
        async def mock_stream():
            chunks = ["This ", "is ", "streaming ", "response"]
            for chunk_text in chunks:
                chunk = Mock()
                chunk.choices = [Mock(delta=Mock(content=chunk_text))]
                yield chunk
        
        with patch.object(generator_service_instance, '_call_llm', return_value=mock_stream()):
            result = []
            async for token in generator_service_instance.stream_generate(
                query="Test query",
                context_chunks=sample_chunks
            ):
                result.append(token)
            
            assert result == ["This ", "is ", "streaming ", "response"]
    
    @pytest.mark.asyncio
    async def test_stream_generate_circuit_breaker_error(self, generator_service_instance, sample_chunks):
        """Test streaming when circuit breaker is open"""
        with patch.object(generator_service_instance._get_circuit_breaker("gpt-4o-mini"), 
                         'call_async') as mock_cb:
            mock_cb.side_effect = CircuitBreakerError("Circuit open")
            
            with pytest.raises(GenerationError) as exc_info:
                async for _ in generator_service_instance.stream_generate(
                    query="Test query",
                    context_chunks=sample_chunks,
                    model_name="gpt-4o-mini"
                ):
                    pass
            
            assert "temporarily unavailable" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_stream_generate_api_error(self, generator_service_instance, sample_chunks):
        """Test streaming with API error"""
        with patch.object(generator_service_instance, '_call_llm') as mock_llm:
            mock_llm.side_effect = Exception("Streaming error")
            
            with pytest.raises(GenerationError) as exc_info:
                async for _ in generator_service_instance.stream_generate(
                    query="Test query",
                    context_chunks=sample_chunks
                ):
                    pass
            
            assert "Failed to stream response" in str(exc_info.value)
    
    def test_validate_model(self, generator_service_instance):
        """Test model validation"""
        assert generator_service_instance.validate_model("gpt-4o-mini") is True
        assert generator_service_instance.validate_model("gpt-4o") is True
        assert generator_service_instance.validate_model("claude-3-haiku-20240307") is True
        assert generator_service_instance.validate_model("unknown-model") is False
    
    def test_get_available_models(self, generator_service_instance):
        """Test getting available models"""
        models = generator_service_instance.get_available_models()
        
        assert "gpt-4o-mini" in models
        assert "gpt-4o" in models
        assert "claude-3-haiku-20240307" in models
        assert len(models) == 4
    
    def test_get_circuit_breaker_status(self, generator_service_instance):
        """Test circuit breaker status retrieval"""
        with patch.object(generator_service_instance.circuit_breakers["openai"], 
                         'get_stats') as mock_stats:
            mock_stats.return_value = {"state": "closed", "total_calls": 100}
            
            status = generator_service_instance.get_circuit_breaker_status()
            
            assert "openai" in status
            assert status["openai"]["state"] == "closed"
    
    @pytest.mark.asyncio
    async def test_generate_with_custom_temperature(self, generator_service_instance, sample_chunks):
        """Test generation with custom temperature"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = Mock(total_tokens=100)
        
        with patch.object(generator_service_instance, '_call_llm') as mock_llm:
            mock_llm.return_value = mock_response
            
            await generator_service_instance.generate(
                query="Test",
                context_chunks=sample_chunks,
                temperature=0.2
            )
            
            # Check that custom temperature was used
            call_args = mock_llm.call_args[1]
            assert call_args["temperature"] == 0.2
    
    @pytest.mark.asyncio
    async def test_generate_with_no_usage_data(self, generator_service_instance, sample_chunks):
        """Test generation when response has no usage data"""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = None
        
        with patch.object(generator_service_instance, '_call_llm', return_value=mock_response):
            response = await generator_service_instance.generate(
                query="Test",
                context_chunks=sample_chunks
            )
            
            assert response.token_count == 0


class TestGeneratorIntegration:
    """Integration tests for generator service"""
    
    @pytest.mark.asyncio
    async def test_full_generation_flow(self):
        """Test complete generation flow"""
        # Create service
        service = GeneratorService()
        
        # Create test chunks
        chunks = [
            RetrievedChunk(
                document_id="doc1",
                document_title="RAG Guide",
                chunk_id="chunk1",
                content="RAG combines retrieval with generation.",
                chunk_index=0,
                similarity_score=0.95
            )
        ]
        
        # Mock the entire LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="RAG is a technique..."))]
        mock_response.usage = Mock(total_tokens=50)
        
        with patch('litellm.acompletion', return_value=mock_response):
            response = await service.generate(
                query="What is RAG?",
                context_chunks=chunks,
                model_name="gpt-4o-mini"
            )
            
            assert response.text == "RAG is a technique..."
            assert response.model_name == "gpt-4o-mini"
            assert response.token_count == 50
            assert response.metadata["context_chunks_used"] == 1
    
    @pytest.mark.asyncio
    async def test_streaming_integration(self):
        """Test streaming generation integration"""
        service = GeneratorService()
        chunks = []
        
        # Create async generator for streaming
        async def mock_stream():
            tokens = ["Hello", " ", "world"]
            for token in tokens:
                chunk = Mock()
                chunk.choices = [Mock(delta=Mock(content=token))]
                yield chunk
        
        with patch('litellm.acompletion', return_value=mock_stream()):
            collected = []
            async for token in service.stream_generate(
                query="Test",
                context_chunks=chunks,
                model_name="gpt-4o-mini"
            ):
                collected.append(token)
            
            assert collected == ["Hello", " ", "world"]