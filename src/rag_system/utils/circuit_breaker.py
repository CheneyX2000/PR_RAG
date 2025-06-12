# src/rag_system/utils/circuit_breaker.py
"""
Circuit breaker implementation for resilient service calls.
Prevents cascading failures by stopping calls to failing services.
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Callable, TypeVar, Dict, Any, Union
import asyncio
from functools import wraps
import time
from dataclasses import dataclass, field

from ..utils.monitoring import logger, error_counter
from ..utils.exceptions import RAGException


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests allowed
    OPEN = "open"          # Service is failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    expected_exceptions: tuple = (Exception,)
    success_threshold: int = 2  # successes needed in HALF_OPEN to close
    timeout: Optional[float] = None  # Call timeout in seconds
    exclude_exceptions: tuple = ()  # Exceptions that don't trigger the breaker


@dataclass
class CircuitBreakerStats:
    """Statistics for monitoring circuit breaker behavior"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: list = field(default_factory=list)


class CircuitBreakerError(RAGException):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Usage:
        breaker = CircuitBreaker(name="openai", failure_threshold=3)
        
        # Sync function
        result = breaker.call(some_function, arg1, arg2)
        
        # Async function
        result = await breaker.call_async(some_async_function, arg1, arg2)
        
        # As decorator
        @breaker.protect
        def protected_function():
            pass
    """
    
    _instances: Dict[str, 'CircuitBreaker'] = {}
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change = datetime.now()
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
        # Register instance
        self._instances[name] = self
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)"""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)"""
        return self._state == CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from OPEN state"""
        if self._state != CircuitState.OPEN:
            return False
        
        if not self._last_failure_time:
            return True
        
        return datetime.now() - self._last_failure_time > timedelta(
            seconds=self.config.recovery_timeout
        )
    
    async def _record_success(self):
        """Record a successful call"""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.successful_calls += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' closed after successful recovery"
                    )
    
    async def _record_failure(self, exception: Exception):
        """Record a failed call"""
        async with self._lock:
            self._stats.total_calls += 1
            self._stats.failed_calls += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = datetime.now()
            self._last_failure_time = datetime.now()
            
            # Record error metric
            error_counter.labels(
                error_type=type(exception).__name__,
                operation=f"circuit_breaker_{self.name}"
            ).inc()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit breaker '{self.name}' reopened due to failure in HALF_OPEN state"
                )
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
                    )
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()
        self._stats.state_changes.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": self._last_state_change
        })
        
        logger.info(
            f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}"
        )
    
    def _check_state(self):
        """Check and potentially update circuit state"""
        if self._should_attempt_reset():
            self._transition_to(CircuitState.HALF_OPEN)
            self._success_count = 0
            logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """
        Call an async function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises an exception
        """
        # Check state before attempting call
        self._check_state()
        
        if self._state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN. Service is unavailable."
            )
        
        start_time = time.time()
        
        try:
            # Apply timeout if configured
            if self.config.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)
            
            await self._record_success()
            
            # Log slow calls
            duration = time.time() - start_time
            if duration > 1.0:  # Warn for calls taking more than 1 second
                logger.warning(
                    f"Slow call through circuit breaker '{self.name}': {duration:.2f}s"
                )
            
            return result
            
        except self.config.exclude_exceptions:
            # These exceptions don't trigger the circuit breaker
            raise
            
        except self.config.expected_exceptions as e:
            await self._record_failure(e)
            raise
            
        except Exception as e:
            # Unexpected exception - record but don't trigger breaker
            logger.error(
                f"Unexpected exception in circuit breaker '{self.name}': {type(e).__name__}"
            )
            raise
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Call a sync function with circuit breaker protection.
        
        Args:
            func: Sync function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If func raises an exception
        """
        # For sync functions, we need to run in an event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def _async_wrapper():
            return func(*args, **kwargs)
        
        return loop.run_until_complete(self.call_async(_async_wrapper))
    
    def protect(self, func: Callable):
        """
        Decorator to protect a function with circuit breaker.
        
        Usage:
            @circuit_breaker.protect
            async def my_function():
                pass
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self._stats.total_calls,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "rejected_calls": self._stats.rejected_calls,
            "failure_rate": (
                self._stats.failed_calls / self._stats.total_calls
                if self._stats.total_calls > 0 else 0
            ),
            "consecutive_failures": self._stats.consecutive_failures,
            "consecutive_successes": self._stats.consecutive_successes,
            "last_failure": (
                self._stats.last_failure_time.isoformat()
                if self._stats.last_failure_time else None
            ),
            "last_success": (
                self._stats.last_success_time.isoformat()
                if self._stats.last_success_time else None
            ),
            "state_changes": self._stats.state_changes[-10:]  # Last 10 changes
        }
    
    def reset(self):
        """Manually reset the circuit breaker"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    @classmethod
    def get_instance(cls, name: str) -> Optional['CircuitBreaker']:
        """Get a circuit breaker instance by name"""
        return cls._instances.get(name)
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: breaker.get_stats()
            for name, breaker in cls._instances.items()
        }


# Pre-configured circuit breakers for common services
class CircuitBreakers:
    """Pre-configured circuit breakers for different services"""
    
    # OpenAI circuit breaker
    openai = CircuitBreaker(
        name="openai",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exceptions=(Exception,),
            timeout=30.0
        )
    )
    
    # Database circuit breaker
    database = CircuitBreaker(
        name="database",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exceptions=(Exception,),
            timeout=10.0
        )
    )
    
    # Redis circuit breaker
    redis = CircuitBreaker(
        name="redis",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=20,
            expected_exceptions=(Exception,),
            timeout=5.0
        )
    )
    
    # External API circuit breaker (generic)
    external_api = CircuitBreaker(
        name="external_api",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=45,
            expected_exceptions=(Exception,),
            timeout=15.0
        )
    )


# Convenience function for creating custom circuit breakers
def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    timeout: Optional[float] = None
) -> CircuitBreaker:
    """
    Create a custom circuit breaker with specified configuration.
    
    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        timeout: Call timeout in seconds
        
    Returns:
        Configured CircuitBreaker instance
    """
    return CircuitBreaker(
        name=name,
        config=CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            timeout=timeout
        )
    )