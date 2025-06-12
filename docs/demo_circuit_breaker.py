# docs/demo_circuit_breaker.py
"""
Demonstration of circuit breaker functionality in the RAG system.
This script simulates various failure scenarios to show how circuit breakers protect the system.
"""

import asyncio
import random
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.rag_system.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError


class ExternalService:
    """Simulated external service that can fail"""
    
    def __init__(self, name: str, failure_rate: float = 0.3):
        self.name = name
        self.failure_rate = failure_rate
        self.call_count = 0
        self.is_down = False
    
    async def call(self, data: str) -> str:
        """Simulate an API call that might fail"""
        self.call_count += 1
        
        # Simulate network delay
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Simulate failures
        if self.is_down or random.random() < self.failure_rate:
            raise Exception(f"{self.name} service error: Connection failed")
        
        return f"{self.name} processed: {data}"
    
    def simulate_outage(self):
        """Simulate service going down"""
        self.is_down = True
        print(f"❌ {self.name} is now DOWN")
    
    def simulate_recovery(self):
        """Simulate service recovery"""
        self.is_down = False
        self.failure_rate = 0.1  # Lower failure rate after recovery
        print(f"✅ {self.name} has RECOVERED")


async def demo_circuit_breaker():
    """Demonstrate circuit breaker behavior"""
    print("🔧 Circuit Breaker Demo")
    print("=" * 50)
    
    # Create services
    openai_service = ExternalService("OpenAI", failure_rate=0.2)
    database_service = ExternalService("Database", failure_rate=0.1)
    
    # Create circuit breakers
    openai_breaker = CircuitBreaker(
        name="openai_demo",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=5,
            success_threshold=2
        )
    )
    
    db_breaker = CircuitBreaker(
        name="database_demo",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=10,
            success_threshold=3
        )
    )
    
    print("\n📊 Initial State:")
    print(f"OpenAI Circuit: {openai_breaker.state.value}")
    print(f"Database Circuit: {db_breaker.state.value}")
    
    # Phase 1: Normal operation with occasional failures
    print("\n🔵 Phase 1: Normal Operation (30 requests)")
    print("-" * 30)
    
    for i in range(30):
        # OpenAI call
        try:
            result = await openai_breaker.call_async(
                openai_service.call,
                f"Query {i}"
            )
            print(f"✅ Request {i}: {result}")
        except CircuitBreakerError as e:
            print(f"🚫 Request {i}: Circuit breaker OPEN - {e}")
        except Exception as e:
            print(f"⚠️  Request {i}: Service error - {e}")
        
        # Small delay between requests
        await asyncio.sleep(0.2)
        
        # Show circuit state every 10 requests
        if (i + 1) % 10 == 0:
            stats = openai_breaker.get_stats()
            print(f"\n📈 Stats after {i+1} requests:")
            print(f"   State: {stats['state']}")
            print(f"   Success rate: {1 - stats['failure_rate']:.1%}")
            print(f"   Consecutive failures: {stats['consecutive_failures']}")
            print()
    
    # Phase 2: Service outage
    print("\n🔴 Phase 2: Service Outage")
    print("-" * 30)
    
    openai_service.simulate_outage()
    
    for i in range(10):
        try:
            result = await openai_breaker.call_async(
                openai_service.call,
                f"Query {30 + i}"
            )
            print(f"✅ Request {30 + i}: {result}")
        except CircuitBreakerError as e:
            print(f"🚫 Request {30 + i}: Circuit breaker protecting system")
        except Exception as e:
            print(f"⚠️  Request {30 + i}: Service error - {e}")
        
        await asyncio.sleep(0.2)
    
    print(f"\n📊 Circuit State: {openai_breaker.state.value}")
    print("   Circuit breaker is preventing cascading failures!")
    
    # Phase 3: Wait for recovery timeout
    print("\n⏳ Phase 3: Waiting for Recovery Timeout (5 seconds)")
    print("-" * 30)
    
    await asyncio.sleep(6)
    openai_service.simulate_recovery()
    
    # Phase 4: Recovery attempts
    print("\n🟡 Phase 4: Recovery Attempts")
    print("-" * 30)
    
    for i in range(10):
        try:
            result = await openai_breaker.call_async(
                openai_service.call,
                f"Recovery test {i}"
            )
            print(f"✅ Recovery test {i}: SUCCESS - {result}")
        except CircuitBreakerError as e:
            print(f"🚫 Recovery test {i}: Still in circuit breaker")
        except Exception as e:
            print(f"⚠️  Recovery test {i}: Service error - {e}")
        
        await asyncio.sleep(0.3)
        
        # Check if circuit closed
        if openai_breaker.state.value == "closed":
            print("\n🎉 Circuit breaker has CLOSED - Service recovered!")
            break
    
    # Final statistics
    print("\n📊 Final Statistics:")
    print("=" * 50)
    
    for name, breaker in [("OpenAI", openai_breaker), ("Database", db_breaker)]:
        stats = breaker.get_stats()
        print(f"\n{name} Circuit Breaker:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Successful: {stats['successful_calls']}")
        print(f"  Failed: {stats['failed_calls']}")
        print(f"  Rejected: {stats['rejected_calls']}")
        print(f"  Final state: {stats['state']}")
        print(f"  State changes: {len(stats['state_changes'])}")
        
        if stats['state_changes']:
            print(f"  Last state change: {stats['state_changes'][-1]}")


async def demo_multi_service():
    """Demonstrate multiple services with circuit breakers"""
    print("\n\n🔄 Multi-Service Circuit Breaker Demo")
    print("=" * 50)
    
    # Simulate a more complex scenario with multiple services
    services = {
        "embedding": ExternalService("Embedding API", failure_rate=0.15),
        "database": ExternalService("Database", failure_rate=0.05),
        "llm": ExternalService("LLM API", failure_rate=0.25),
    }
    
    breakers = {
        name: CircuitBreaker(
            name=f"{name}_breaker",
            config=CircuitBreakerConfig(
                failure_threshold=3 if name == "llm" else 5,
                recovery_timeout=5,
                success_threshold=2
            )
        )
        for name in services
    }
    
    print("\n🚀 Simulating RAG system with multiple services...")
    
    async def process_rag_query(query_id: int):
        """Simulate a complete RAG query"""
        results = {}
        
        # Step 1: Generate embeddings
        try:
            embedding = await breakers["embedding"].call_async(
                services["embedding"].call,
                f"Query {query_id}"
            )
            results["embedding"] = "✅"
        except (CircuitBreakerError, Exception):
            results["embedding"] = "❌"
            return results  # Can't continue without embeddings
        
        # Step 2: Database search
        try:
            docs = await breakers["database"].call_async(
                services["database"].call,
                f"Search {query_id}"
            )
            results["database"] = "✅"
        except (CircuitBreakerError, Exception):
            results["database"] = "❌"
            return results  # Can't continue without documents
        
        # Step 3: LLM generation
        try:
            response = await breakers["llm"].call_async(
                services["llm"].call,
                f"Generate {query_id}"
            )
            results["llm"] = "✅"
        except (CircuitBreakerError, Exception):
            results["llm"] = "❌"
        
        return results
    
    # Process multiple queries
    for i in range(20):
        print(f"\nQuery {i}: ", end="")
        results = await process_rag_query(i)
        
        # Show results
        status_str = " → ".join([f"{k}:{v}" for k, v in results.items()])
        print(status_str)
        
        # Show breaker states every 5 queries
        if (i + 1) % 5 == 0:
            print("\n  Circuit States:")
            for name, breaker in breakers.items():
                print(f"    {name}: {breaker.state.value}")
        
        await asyncio.sleep(0.5)
    
    # Show final statistics
    print("\n\n📊 Service Statistics:")
    all_stats = CircuitBreaker.get_all_stats()
    
    for name, stats in all_stats.items():
        if "_breaker" in name:
            service_name = name.replace("_breaker", "")
            print(f"\n{service_name.capitalize()}:")
            print(f"  Success rate: {1 - stats['failure_rate']:.1%}")
            print(f"  State: {stats['state']}")
            print(f"  Calls: {stats['total_calls']} total, {stats['rejected_calls']} rejected")


if __name__ == "__main__":
    print("🎮 RAG System Circuit Breaker Demonstration")
    print("This demo shows how circuit breakers protect the system from cascading failures")
    print()
    
    # Run the demos
    asyncio.run(demo_circuit_breaker())
    asyncio.run(demo_multi_service())
    
    print("\n\n✅ Demo completed!")
    print("Circuit breakers help maintain system stability by:")
    print("  • Preventing cascading failures")
    print("  • Giving failed services time to recover")
    print("  • Providing fallback behavior")
    print("  • Monitoring service health")