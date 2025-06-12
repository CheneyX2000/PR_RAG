# demo_reranking.py
"""
Demonstration of the reranking functionality in the RAG system.
Shows how reranking can significantly improve retrieval quality.
"""

import asyncio
import httpx
from typing import List, Dict
import json
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# API base URL
BASE_URL = "http://localhost:8000/api/v1"


async def setup_demo_documents():
    """Ingest some demo documents about RAG and vector databases"""
    documents = [
        {
            "title": "Introduction to RAG",
            "content": """
            Retrieval-Augmented Generation (RAG) is a powerful technique that combines 
            the capabilities of large language models with external knowledge retrieval. 
            RAG systems work by first retrieving relevant documents from a knowledge base, 
            then using these documents as context for generating responses. This approach 
            allows LLMs to access up-to-date information and provide more accurate, 
            grounded responses without requiring constant retraining.
            """
        },
        {
            "title": "Vector Similarity Search",
            "content": """
            Vector similarity search is the foundation of modern RAG systems. It converts 
            text into high-dimensional vectors (embeddings) that capture semantic meaning. 
            Similar concepts are represented by vectors that are close together in the 
            vector space. Common similarity metrics include cosine similarity, Euclidean 
            distance, and dot product. The choice of embedding model and similarity metric 
            significantly impacts retrieval quality.
            """
        },
        {
            "title": "Reranking in Information Retrieval",
            "content": """
            Reranking is a crucial technique for improving search quality in RAG systems. 
            While initial retrieval using vector similarity is fast and scalable, it may 
            not always capture the most relevant documents. Reranking uses more sophisticated 
            models, typically cross-encoders, to score the relevance of query-document pairs. 
            This two-stage approach balances efficiency with accuracy, retrieving a larger 
            set of candidates quickly, then carefully reordering them.
            """
        },
        {
            "title": "Cross-Encoder Models",
            "content": """
            Cross-encoder models are transformer-based architectures that take both the 
            query and document as input, processing them together to produce a relevance 
            score. Unlike bi-encoders used in vector search, cross-encoders can capture 
            fine-grained interactions between query and document terms. Popular cross-encoder 
            models include MS MARCO MiniLM variants, which are fine-tuned on large-scale 
            passage ranking datasets. These models significantly outperform vector similarity 
            alone but are more computationally expensive.
            """
        },
        {
            "title": "Building Production RAG Systems",
            "content": """
            Production RAG systems require careful consideration of performance, accuracy, 
            and cost. Key components include efficient vector databases like PgVector, 
            proper chunking strategies, and hybrid retrieval approaches. Implementing 
            reranking is essential for high-quality results. Other considerations include 
            caching, monitoring, error handling, and fallback mechanisms. The system should 
            be designed to handle failures gracefully and scale with increasing data and 
            query volumes.
            """
        }
    ]
    
    console.print("[bold blue]Ingesting demo documents...[/bold blue]")
    
    async with httpx.AsyncClient() as client:
        for doc in track(documents, description="Ingesting documents"):
            try:
                response = await client.post(
                    f"{BASE_URL}/ingest",
                    json=doc
                )
                if response.status_code == 200:
                    console.print(f"✅ Ingested: {doc['title']}")
                else:
                    console.print(f"❌ Failed to ingest: {doc['title']}")
            except Exception as e:
                console.print(f"❌ Error: {e}")


async def demo_basic_search(query: str):
    """Demonstrate basic search without reranking"""
    console.print(f"\n[bold yellow]Basic Search (No Reranking)[/bold yellow]")
    console.print(f"Query: '{query}'")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "top_k": 5,
                "rerank": False
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            table = Table(title="Search Results (Vector Similarity Only)")
            table.add_column("Rank", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Similarity Score", style="yellow")
            table.add_column("Preview", style="white")
            
            for i, result in enumerate(data["results"], 1):
                table.add_row(
                    str(i),
                    result["document_title"],
                    f"{result['similarity_score']:.4f}",
                    result["content"][:80] + "..."
                )
            
            console.print(table)
            return data["results"]
        else:
            console.print(f"[red]Search failed: {response.status_code}[/red]")
            return []


async def demo_reranked_search(query: str, model: str = "ms-marco-MiniLM-L-6-v2"):
    """Demonstrate search with reranking"""
    console.print(f"\n[bold yellow]Reranked Search[/bold yellow]")
    console.print(f"Query: '{query}'")
    console.print(f"Reranking Model: {model}")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "top_k": 5,
                "rerank": True,
                "rerank_model": model
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            table = Table(title="Reranked Search Results")
            table.add_column("Rank", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Similarity", style="yellow")
            table.add_column("Rerank Score", style="magenta")
            table.add_column("Preview", style="white")
            
            for i, result in enumerate(data["results"], 1):
                table.add_row(
                    str(i),
                    result["document_title"],
                    f"{result['similarity_score']:.4f}",
                    f"{result['rerank_score']:.4f}" if result.get('rerank_score') else "N/A",
                    result["content"][:80] + "..."
                )
            
            console.print(table)
            return data["results"]
        else:
            console.print(f"[red]Search failed: {response.status_code}[/red]")
            return []


async def demo_comparison(query: str):
    """Compare search results with and without reranking"""
    console.print(f"\n[bold yellow]Comparing Search Methods[/bold yellow]")
    console.print(f"Query: '{query}'")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/search/compare",
            json={
                "query": query,
                "top_k": 5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Show side-by-side comparison
            table = Table(title="Search Results Comparison")
            table.add_column("Rank", style="cyan")
            table.add_column("Without Reranking", style="yellow")
            table.add_column("With Reranking", style="green")
            table.add_column("Rank Change", style="magenta")
            
            for i in range(5):
                without = data["without_reranking"][i] if i < len(data["without_reranking"]) else None
                with_rerank = data["with_reranking"][i] if i < len(data["with_reranking"]) else None
                
                # Find rank change
                rank_change = ""
                if with_rerank and without:
                    doc_id = with_rerank["document_id"]
                    if doc_id in data["rank_changes"]:
                        change = data["rank_changes"][doc_id]["change"]
                        if change > 0:
                            rank_change = f"↑{change}"
                        elif change < 0:
                            rank_change = f"↓{abs(change)}"
                        else:
                            rank_change = "="
                
                table.add_row(
                    str(i + 1),
                    without["title"] if without else "-",
                    with_rerank["title"] if with_rerank else "-",
                    rank_change
                )
            
            console.print(table)
            
            # Show summary
            summary = data["summary"]
            console.print("\n[bold]Summary:[/bold]")
            console.print(f"  Documents reordered: {summary['documents_reordered']}")
            console.print(f"  New documents in top-5: {summary['new_documents']}")
            console.print(f"  Documents removed from top-5: {summary['removed_documents']}")
            
        else:
            console.print(f"[red]Comparison failed: {response.status_code}[/red]")


async def demo_reranking_models():
    """Show available reranking models"""
    console.print("\n[bold yellow]Available Reranking Models[/bold yellow]")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/reranking/models")
        
        if response.status_code == 200:
            models = response.json()
            
            table = Table(title="Reranking Models")
            table.add_column("Model Name", style="cyan")
            table.add_column("Batch Size", style="yellow")
            table.add_column("Max Length", style="green")
            table.add_column("Default", style="magenta")
            table.add_column("Loaded", style="blue")
            
            for name, info in models.items():
                table.add_row(
                    name,
                    str(info["batch_size"]),
                    str(info["max_length"]),
                    "✓" if info["is_default"] else "",
                    "✓" if info["is_loaded"] else ""
                )
            
            console.print(table)
        else:
            console.print(f"[red]Failed to get models: {response.status_code}[/red]")


async def run_demo():
    """Run the complete reranking demo"""
    console.print("[bold green]RAG Reranking Demo[/bold green]")
    console.print("=" * 50)
    
    # Setup documents (skip if already ingested)
    console.print("\n[bold]Setup[/bold]")
    setup = console.input("Do you want to ingest demo documents? (y/n): ")
    if setup.lower() == 'y':
        await setup_demo_documents()
    
    # Show available models
    await demo_reranking_models()
    
    # Demo queries
    queries = [
        "What is reranking and why is it important?",
        "How do cross-encoder models work?",
        "What are the key components of a production RAG system?",
        "Explain vector similarity search"
    ]
    
    console.print("\n[bold]Running Search Demos[/bold]")
    console.print("We'll test several queries with and without reranking")
    
    for query in queries:
        console.print("\n" + "="*80)
        
        # Basic search
        basic_results = await demo_basic_search(query)
        
        # Reranked search
        reranked_results = await demo_reranked_search(query)
        
        # Show comparison
        await demo_comparison(query)
        
        # Wait for user input
        console.input("\nPress Enter to continue...")
    
    # Test different reranking models
    console.print("\n[bold]Testing Different Reranking Models[/bold]")
    test_query = "What makes a good RAG system?"
    
    models = [
        "ms-marco-TinyBERT-L-2-v2",  # Fastest
        "ms-marco-MiniLM-L-6-v2",     # Balanced
        "ms-marco-MiniLM-L-12-v2"     # Most accurate
    ]
    
    for model in models:
        await demo_reranked_search(test_query, model)
        console.input("\nPress Enter to continue...")
    
    console.print("\n[bold green]Demo Complete![/bold green]")
    console.print("\nKey Takeaways:")
    console.print("1. Reranking can significantly improve search relevance")
    console.print("2. Cross-encoders capture query-document interactions better than embeddings alone")
    console.print("3. Different reranking models offer trade-offs between speed and accuracy")
    console.print("4. The two-stage retrieve-then-rerank approach balances efficiency and quality")


if __name__ == "__main__":
    # Check if rich is installed
    try:
        from rich.console import Console
    except ImportError:
        print("Please install rich: pip install rich")
        exit(1)
    
    asyncio.run(run_demo())