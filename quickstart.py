# quickstart.py
import asyncio
import os
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.rag_system.db.pgvector import init_database
from src.rag_system.services.ingestion import ingestion_service
from src.rag_system.services.retriever import retriever

async def main():
    print("🚀 RAG System Quick Start")
    print("-" * 50)
    
    # 1. Initialize database
    print("1. Initializing database...")
    try:
        await init_database()
        print("   ✅ Database initialized successfully!")
    except Exception as e:
        print(f"   ❌ Database initialization failed: {e}")
        print("   Make sure PostgreSQL is running with: docker-compose -f docker-compose.dev.yml up -d")
        return
    
    # 2. Ingest sample documents
    print("\n2. Ingesting sample documents...")
    
    sample_documents = [
        {
            "title": "Introduction to RAG",
            "content": """
            Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models 
            with external knowledge retrieval. It works by first retrieving relevant documents from a knowledge base, 
            then using these documents as context for generating responses. This approach allows LLMs to access 
            up-to-date information and provide more accurate, grounded responses.
            
            The key components of a RAG system include:
            1. Document ingestion and preprocessing
            2. Embedding generation for semantic search
            3. Vector database for efficient similarity search
            4. Retrieval mechanism to find relevant documents
            5. Generation component that uses retrieved context
            """
        },
        {
            "title": "PgVector Basics",
            "content": """
            PgVector is a PostgreSQL extension that adds support for vector similarity search. It enables storing 
            and querying high-dimensional vectors directly in PostgreSQL. PgVector supports multiple index types 
            including HNSW (Hierarchical Navigable Small World) and IVFFlat for efficient nearest neighbor search.
            
            Key features of PgVector:
            - Native PostgreSQL integration
            - Support for cosine, L2, and inner product distance metrics
            - Efficient indexing with HNSW and IVFFlat
            - ACID compliance and full SQL support
            - Easy to scale and maintain
            """
        },
        {
            "title": "Python Best Practices for RAG",
            "content": """
            When building RAG systems in Python, following best practices ensures maintainability and performance:
            
            1. Use async/await for I/O operations to handle concurrent requests efficiently
            2. Implement proper error handling and retry mechanisms
            3. Use connection pooling for database connections
            4. Batch operations when processing multiple documents
            5. Implement caching for frequently accessed data
            6. Use type hints and Pydantic for data validation
            7. Structure your project with clear separation of concerns
            8. Write comprehensive tests for all components
            """
        }
    ]
    
    for doc in sample_documents:
        try:
            doc_id = await ingestion_service.ingest_document(**doc)
            print(f"   ✅ Ingested: {doc['title']} (ID: {doc_id})")
        except Exception as e:
            print(f"   ❌ Failed to ingest {doc['title']}: {e}")
    
    # 3. Test retrieval
    print("\n3. Testing retrieval...")
    
    test_queries = [
        "What is RAG and how does it work?",
        "Tell me about PgVector indexing options",
        "What are Python best practices for building RAG systems?",
        "How to handle errors in async Python?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        try:
            results = await retriever.search(query, top_k=3)
            print(f"   Found {len(results)} relevant chunks:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.document_title} (similarity: {result.similarity_score:.3f})")
                print(f"      Preview: {result.content[:100]}...")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    print("\n" + "="*50)
    print("✅ Quick start completed!")
    print("\nNext steps:")
    print("1. Start the API server: uvicorn src.rag_system.main:app --reload")
    print("2. Visit http://localhost:8000/docs for interactive API documentation")
    print("3. Try ingesting your own documents via the API")

if __name__ == "__main__":
    asyncio.run(main())