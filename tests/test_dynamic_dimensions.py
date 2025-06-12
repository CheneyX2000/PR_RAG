# tests/test_dynamic_dimensions.py
"""
Test script to verify dynamic dimension support is working correctly
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.rag_system.db.pgvector import db
from src.rag_system.services.embeddings import embedding_service
from src.rag_system.services.ingestion import ingestion_service
from src.rag_system.services.retriever import retriever


async def test_embedding_models():
    """Test that different embedding models work correctly"""
    print("\n🧪 Testing Embedding Models")
    print("=" * 50)
    
    test_text = "This is a test sentence for embeddings."
    
    models_to_test = [
        ("text-embedding-ada-002", 1536),
        ("all-MiniLM-L6-v2", 384),
    ]
    
    for model_name, expected_dim in models_to_test:
        try:
            print(f"\n📊 Testing {model_name}...")
            
            # Get model dimension
            dimension = embedding_service.get_model_dimension(model_name)
            print(f"  Expected dimension: {expected_dim}")
            print(f"  Actual dimension: {dimension}")
            assert dimension == expected_dim, f"Dimension mismatch for {model_name}"
            
            # Generate embedding
            embedding, dim = await embedding_service.embed_text(test_text, model_name=model_name)
            print(f"  Generated embedding length: {len(embedding)}")
            assert len(embedding) == expected_dim, f"Embedding size mismatch for {model_name}"
            
            print(f"  ✅ {model_name} working correctly!")
            
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            return False
    
    return True


async def test_document_ingestion():
    """Test ingesting documents with different models"""
    print("\n🧪 Testing Document Ingestion")
    print("=" * 50)
    
    test_documents = [
        {
            "title": "Test Doc 1 - OpenAI",
            "content": "This document tests OpenAI embeddings with 1536 dimensions.",
            "model": "text-embedding-ada-002"
        },
        {
            "title": "Test Doc 2 - MiniLM",
            "content": "This document tests MiniLM embeddings with 384 dimensions.",
            "model": "all-MiniLM-L6-v2"
        }
    ]
    
    document_ids = []
    
    for doc in test_documents:
        try:
            print(f"\n📄 Ingesting '{doc['title']}' with {doc['model']}...")
            
            doc_id = await ingestion_service.ingest_document(
                title=doc["title"],
                content=doc["content"],
                embedding_model=doc["model"]
            )
            
            document_ids.append(doc_id)
            print(f"  ✅ Document ingested with ID: {doc_id}")
            
            # Check embedding status
            status = await ingestion_service.get_document_embedding_status(doc_id)
            print(f"  📊 Embedding status:")
            for model, info in status["embeddings"].items():
                print(f"     - {model}: {info['count']} chunks, dimension={info['dimension']}")
                
        except Exception as e:
            print(f"  ❌ Failed to ingest document: {e}")
            return False, []
    
    return True, document_ids


async def test_retrieval():
    """Test retrieval with different models"""
    print("\n🧪 Testing Retrieval")
    print("=" * 50)
    
    test_query = "Tell me about embeddings and dimensions"
    
    models_to_search = ["text-embedding-ada-002", "all-MiniLM-L6-v2"]
    
    for model_name in models_to_search:
        try:
            print(f"\n🔍 Searching with {model_name}...")
            
            results = await retriever.search(
                query=test_query,
                model_name=model_name,
                top_k=3
            )
            
            print(f"  Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.document_title} (score: {result.similarity_score:.3f})")
                
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
            return False
    
    # Test multi-model search
    try:
        print(f"\n🔍 Testing multi-model search...")
        
        results = await retriever.search_multi_model(
            query=test_query,
            model_names=models_to_search,
            top_k=3,
            aggregation="union"
        )
        
        print(f"  Found {len(results)} aggregated results")
        
    except Exception as e:
        print(f"  ❌ Multi-model search failed: {e}")
        return False
    
    return True


async def test_dimension_info():
    """Test database dimension information"""
    print("\n🧪 Testing Dimension Information")
    print("=" * 50)
    
    try:
        # Get supported dimensions
        dimensions = await db.get_supported_dimensions()
        print(f"\n📏 Supported dimensions in database: {dimensions}")
        
        # Get available models for search
        models = await retriever.get_available_models_for_search()
        print(f"\n🔍 Models available for search:")
        for model in models:
            print(f"  - {model['model_name']}: "
                  f"dimension={model['dimension']}, "
                  f"embeddings={model['embedding_count']}, "
                  f"current={'✓' if model['is_current'] else '✗'}")
                  
    except Exception as e:
        print(f"  ❌ Failed to get dimension info: {e}")
        return False
    
    return True


async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Dynamic Dimension Tests")
    print("=" * 50)
    
    try:
        # Initialize database
        print("Initializing database connection...")
        await db.initialize()
        
        # Run tests
        results = {}
        
        # Test 1: Embedding models
        results["embeddings"] = await test_embedding_models()
        
        # Test 2: Document ingestion
        ingestion_success, doc_ids = await test_document_ingestion()
        results["ingestion"] = ingestion_success
        
        # Test 3: Retrieval (only if ingestion succeeded)
        if ingestion_success:
            results["retrieval"] = await test_retrieval()
        else:
            results["retrieval"] = False
            
        # Test 4: Dimension info
        results["dimension_info"] = await test_dimension_info()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        print("=" * 50)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name.ljust(20)}: {status}")
            if not passed:
                all_passed = False
        
        print("=" * 50)
        
        if all_passed:
            print("🎉 All tests passed! Dynamic dimensions are working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the output above.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        await db.close()


if __name__ == "__main__":
    asyncio.run(run_all_tests())