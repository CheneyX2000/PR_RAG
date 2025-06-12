# tests/test_circuit_breaker.py
"""
Tests for circuit breaker functionality
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from src.rag_system.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerError,
    create_circuit_breaker
)


class TestCircuitBreaker:
    """Test cases for circuit breaker"""
    
    @pytest.fixture
    def breaker(self):
        """Create a test circuit breaker"""
        return CircuitBreaker(
            name="test_breaker",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=5,
                success_threshold=2
            )
        )
    
    @pytest.mark.asyncio
    async def test_initial_state(self, breaker):
        """Test circuit breaker starts in closed state"""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open
    
    @pytest.mark.asyncio
    async def test_successful_calls(self, breaker):
        """Test successful calls don't open the breaker"""
        async def success_func():
            return "success"
        
        # Make multiple successful calls
        for _ in range(5):
            result = await breaker.call_async(success_func)
            assert result == "success"
        
        # Breaker should still be closed
        assert breaker.state == CircuitState.CLOSED
        
        # Check stats
        stats = breaker.get_stats()
        assert stats["successful_calls"] == 5
        assert stats["failed_calls"] == 0
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self, breaker):
        """Test circuit opens after failure threshold"""
        async def failing_func():
            raise Exception("Test failure")
        
        # Make calls up to failure threshold
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(failing_func)
        
        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open
        
        # Further calls should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.call_async(failing_func)
        
        assert "is OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_half_open_state(self, breaker):
        """Test circuit transitions to half-open after timeout"""
        async def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(6)  # Recovery timeout is 5 seconds
        
        # Next call should attempt (half-open state)
        breaker._check_state()
        assert breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_recovery_from_half_open(self, breaker):
        """Test successful recovery from half-open state"""
        call_count = 0
        
        async def variable_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Failure")
            return "success"
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(variable_func)
        
        # Force to half-open
        breaker._transition_to(CircuitState.HALF_OPEN)
        
        # Make successful calls to close the circuit
        for _ in range(2):  # success_threshold is 2
            result = await breaker.call_async(variable_func)
            assert result == "success"
        
        # Circuit should be closed again
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_failure_in_half_open_reopens(self, breaker):
        """Test failure in half-open state reopens the circuit"""
        async def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call_async(failing_func)
        
        # Force to half-open
        breaker._transition_to(CircuitState.HALF_OPEN)
        
        # Failure in half-open should reopen
        with pytest.raises(Exception):
            await breaker.call_async(failing_func)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_timeout_functionality(self):
        """Test call timeout functionality"""
        breaker = CircuitBreaker(
            name="timeout_test",
            config=CircuitBreakerConfig(timeout=0.1)  # 100ms timeout
        )
        
        async def slow_func():
            await asyncio.sleep(1)  # Sleep for 1 second
            return "done"
        
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await breaker.call_async(slow_func)
    
    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Test exceptions that don't trigger the breaker"""
        breaker = CircuitBreaker(
            name="exclude_test",
            config=CircuitBreakerConfig(
                failure_threshold=2,
                exclude_exceptions=(ValueError,)
            )
        )
        
        async def value_error_func():
            raise ValueError("This should not trigger breaker")
        
        # These should not count as failures
        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.call_async(value_error_func)
        
        # Breaker should still be closed
        assert breaker.state == CircuitState.CLOSED
        stats = breaker.get_stats()
        assert stats["failed_calls"] == 0
    
    def test_sync_call(self, breaker):
        """Test synchronous function calls"""
        def sync_func(x, y):
            return x + y
        
        result = breaker.call(sync_func, 2, 3)
        assert result == 5
        
        # Check it recorded the success
        stats = breaker.get_stats()
        assert stats["successful_calls"] == 1
    
    @pytest.mark.asyncio
    async def test_decorator_async(self, breaker):
        """Test circuit breaker as async decorator"""
        call_count = 0
        
        @breaker.protect
        async def protected_async_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Failure")
            return "success"
        
        # First 3 calls fail
        for _ in range(3):
            with pytest.raises(Exception):
                await protected_async_func()
        
        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            await protected_async_func()
    
    def test_decorator_sync(self, breaker):
        """Test circuit breaker as sync decorator"""
        @breaker.protect
        def protected_sync_func(x):
            return x * 2
        
        result = protected_sync_func(5)
        assert result == 10
    
    def test_manual_reset(self, breaker):
        """Test manual reset functionality"""
        # Open the circuit
        breaker._transition_to(CircuitState.OPEN)
        assert breaker.state == CircuitState.OPEN
        
        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
    
    def test_get_instance(self):
        """Test getting circuit breaker instance by name"""
        breaker = CircuitBreaker(name="test_instance")
        
        # Should be able to get it by name
        retrieved = CircuitBreaker.get_instance("test_instance")
        assert retrieved is breaker
        
        # Non-existent should return None
        assert CircuitBreaker.get_instance("non_existent") is None
    
    def test_create_circuit_breaker_helper(self):
        """Test the create_circuit_breaker helper function"""
        breaker = create_circuit_breaker(
            name="helper_test",
            failure_threshold=5,
            recovery_timeout=30,
            timeout=10.0
        )
        
        assert breaker.name == "helper_test"
        assert breaker.config.failure_threshold == 5
        assert breaker.config.recovery_timeout == 30
        assert breaker.config.timeout == 10.0
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, breaker):
        """Test comprehensive statistics tracking"""
        async def variable_func(should_fail=False):
            if should_fail:
                raise Exception("Failure")
            return "success"
        
        # Some successes
        for _ in range(3):
            await breaker.call_async(variable_func, should_fail=False)
        
        # Some failures
        for _ in range(2):
            with pytest.raises(Exception):
                await breaker.call_async(variable_func, should_fail=True)
        
        stats = breaker.get_stats()
        assert stats["total_calls"] == 5
        assert stats["successful_calls"] == 3
        assert stats["failed_calls"] == 2
        assert stats["failure_rate"] == 0.4
        assert stats["consecutive_failures"] == 2
        assert stats["consecutive_successes"] == 0
        assert stats["last_failure"] is not None
        assert stats["last_success"] is not None
        assert len(stats["state_changes"]) > 0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breakers with services"""
    
    @pytest.mark.asyncio
    async def test_embedding_service_circuit_breaker(self, mock_embedding_service):
        """Test circuit breaker integration with embedding service"""
        from src.rag_system.utils.circuit_breaker import CircuitBreakers
        
        # Mock OpenAI to fail
        async def failing_openai_call(*args, **kwargs):
            raise Exception("OpenAI API error")
        
        # Replace the OpenAI circuit breaker call
        original_call = CircuitBreakers.openai.call_async
        CircuitBreakers.openai.call_async = failing_openai_call
        
        try:
            # Reset the breaker first
            CircuitBreakers.openai.reset()
            
            # This should eventually open the circuit
            with pytest.raises(Exception):
                for _ in range(5):
                    await mock_embedding_service.embed_text("test")
        finally:
            # Restore original
            CircuitBreakers.openai.call_async = original_call
            CircuitBreakers.openai.reset()