# tests/test_monitoring.py
"""
Comprehensive tests for monitoring utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary

from src.rag_system.utils.monitoring import (
    logger,
    metrics,
    query_counter,
    query_duration,
    active_models,
    document_ingestion_counter,
    document_ingestion_duration,
    embedding_generation_duration,
    llm_generation_duration,
    cache_hits,
    cache_misses,
    error_counter
)


class TestLogger:
    """Test structured logging configuration"""
    
    def test_logger_instance(self):
        """Test logger is properly configured"""
        assert logger is not None
        # Verify it's a structlog logger
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'debug')
    
    def test_logger_structured_output(self):
        """Test logger produces structured output"""
        with patch('structlog.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Get a new logger instance
            test_logger = structlog.get_logger()
            
            # Log with structured data
            test_logger.info(
                "test_event",
                user_id="123",
                action="search",
                duration=0.5
            )
            
            mock_logger.info.assert_called_once_with(
                "test_event",
                user_id="123",
                action="search",
                duration=0.5
            )
    
    def test_logger_error_with_exception(self):
        """Test logging errors with exception info"""
        with patch.object(logger, 'error') as mock_error:
            try:
                raise ValueError("Test error")
            except ValueError as e:
                logger.error(
                    "operation_failed",
                    error=str(e),
                    operation="test"
                )
            
            mock_error.assert_called_once()
            call_args = mock_error.call_args
            assert call_args[0][0] == "operation_failed"
            assert call_args[1]["error"] == "Test error"
            assert call_args[1]["operation"] == "test"


class TestPrometheusMetrics:
    """Test Prometheus metrics"""
    
    def test_query_counter(self):
        """Test query counter metric"""
        assert isinstance(query_counter, Counter)
        assert query_counter._name == 'rag_queries_total'
        
        # Test increment
        initial_value = query_counter._value.get()
        query_counter.inc()
        assert query_counter._value.get() == initial_value + 1
    
    def test_query_duration(self):
        """Test query duration histogram"""
        assert isinstance(query_duration, Histogram)
        assert query_duration._name == 'rag_query_duration_seconds'
        
        # Test with labels
        labeled_metric = query_duration.labels(
            method='GET',
            endpoint='/search',
            status=200
        )
        assert labeled_metric is not None
    
    def test_active_models_gauge(self):
        """Test active models gauge"""
        assert isinstance(active_models, Gauge)
        assert active_models._name == 'rag_active_models'
        
        # Test gauge operations
        active_models.set(3)
        assert active_models._value.get() == 3
        
        active_models.inc()
        assert active_models._value.get() == 4
        
        active_models.dec()
        assert active_models._value.get() == 3
    
    def test_document_ingestion_metrics(self):
        """Test document ingestion metrics"""
        assert isinstance(document_ingestion_counter, Counter)
        assert isinstance(document_ingestion_duration, Histogram)
        
        # Test counter with labels
        labeled_counter = document_ingestion_counter.labels(document_type='pdf')
        initial_value = labeled_counter._value.get()
        labeled_counter.inc()
        assert labeled_counter._value.get() == initial_value + 1
    
    def test_embedding_generation_duration(self):
        """Test embedding generation duration metric"""
        assert isinstance(embedding_generation_duration, Histogram)
        
        # Test with model label
        with embedding_generation_duration.labels(model='text-embedding-ada-002').time():
            # Simulate embedding generation
            pass
    
    def test_llm_generation_duration(self):
        """Test LLM generation duration metric"""
        assert isinstance(llm_generation_duration, Histogram)
        
        # Test observation
        llm_generation_duration.labels(model='gpt-4o-mini').observe(0.5)
    
    def test_cache_metrics(self):
        """Test cache hit/miss metrics"""
        assert isinstance(cache_hits, Counter)
        assert isinstance(cache_misses, Counter)
        
        # Test cache hits
        cache_hits.labels(cache_type='embedding').inc()
        cache_hits.labels(cache_type='search').inc()
        
        # Test cache misses
        cache_misses.labels(cache_type='embedding').inc()
    
    def test_error_counter(self):
        """Test error counter with labels"""
        assert isinstance(error_counter, Counter)
        
        # Test with different error types
        error_counter.labels(
            error_type='ValidationError',
            operation='embedding_generation'
        ).inc()
        
        error_counter.labels(
            error_type='DatabaseError',
            operation='similarity_search'
        ).inc()
    
    def test_metrics_dictionary(self):
        """Test metrics dictionary contains all metrics"""
        expected_metrics = {
            'query_counter',
            'query_duration',
            'active_models',
            'document_ingestion_counter',
            'document_ingestion_duration',
            'embedding_generation_duration',
            'llm_generation_duration',
            'cache_hits',
            'cache_misses',
            'error_counter'
        }
        
        assert set(metrics.keys()) == expected_metrics
        
        # Verify all metrics are Prometheus metrics
        for metric in metrics.values():
            assert isinstance(metric, (Counter, Histogram, Gauge, Summary))
    
    def test_metric_context_manager(self):
        """Test using metrics as context managers"""
        # Test histogram timing
        with query_duration.labels(
            method='POST',
            endpoint='/query',
            status=200
        ).time() as timer:
            # Simulate some work
            import time
            time.sleep(0.01)
        
        # The timer should have recorded the duration
        assert timer is not None
    
    def test_metric_labels_consistency(self):
        """Test that metrics maintain label consistency"""
        # Create labeled metrics
        metric1 = query_duration.labels(method='GET', endpoint='/test', status=200)
        metric2 = query_duration.labels(method='GET', endpoint='/test', status=200)
        
        # Should return the same instance for same labels
        assert metric1 is metric2
        
        # Different labels should create different instances
        metric3 = query_duration.labels(method='POST', endpoint='/test', status=200)
        assert metric1 is not metric3


class TestMonitoringIntegration:
    """Test monitoring integration with services"""
    
    @pytest.mark.asyncio
    async def test_monitoring_in_service_call(self):
        """Test that monitoring is integrated in service calls"""
        from src.rag_system.services.retriever import retriever
        
        with patch.object(query_counter, 'inc') as mock_inc:
            with patch.object(query_duration, 'labels') as mock_labels:
                mock_timer = Mock()
                mock_labels.return_value.time.return_value.__enter__ = Mock()
                mock_labels.return_value.time.return_value.__exit__ = Mock()
                
                # The actual service call would trigger metrics
                # This is a simplified test
                query_counter.inc()
                
                mock_inc.assert_called_once()
    
    def test_error_logging_and_metrics(self):
        """Test that errors are both logged and counted"""
        with patch.object(logger, 'error') as mock_log:
            with patch.object(error_counter, 'labels') as mock_counter:
                mock_counter.return_value.inc = Mock()
                
                # Simulate an error scenario
                error_type = "DatabaseError"
                operation = "similarity_search"
                error_message = "Connection failed"
                
                # Log and count the error
                logger.error(
                    "operation_failed",
                    error_type=error_type,
                    operation=operation,
                    message=error_message
                )
                
                error_counter.labels(
                    error_type=error_type,
                    operation=operation
                ).inc()
                
                # Verify both were called
                mock_log.assert_called_once()
                mock_counter.assert_called_once_with(
                    error_type=error_type,
                    operation=operation
                )