"""
Session Query Cache Usage Example
==================================

This script demonstrates how to use the in-memory query cache
to avoid redundant searches during the same session.
"""

from ex import BuildingRag

def demo_cache_usage():
    """Demonstrate query caching in action"""
    
    # Initialize RAG system (cache is auto-initialized)
    rag = BuildingRag()
    
    # Setup Neo4j connection
    rag.setup_neo4j_graph()
    
    print("\n" + "="*60)
    print("SESSION QUERY CACHE DEMO")
    print("="*60)
    
    # Example 1: Basic mode - same query twice
    print("\n📝 Example 1: Testing Basic Mode Cache")
    print("-" * 60)
    query1 = "What is machine learning?"
    
    print("\n🔍 First call (will hit database):")
    results1 = rag.handle_query(query1, mode="basic")
    print(f"   Got {len(results1)} results")
    
    print("\n🔍 Second call (should use cache):")
    results2 = rag.handle_query(query1, mode="basic")
    print(f"   Got {len(results2)} results")
    
    # Example 2: Deep mode - more comprehensive search
    print("\n\n📝 Example 2: Testing Deep Mode")
    print("-" * 60)
    query2 = "neural networks architecture"
    
    print("\n🔍 First call (will hit database):")
    results3 = rag.handle_query(query2, mode="deep")
    print(f"   Got {len(results3)} results")
    
    print("\n🔍 Second call (should use cache):")
    results4 = rag.handle_query(query2, mode="deep")
    print(f"   Got {len(results4)} results")
    
    # Example 3: Same query, different modes
    print("\n\n📝 Example 3: Same Query, Different Modes")
    print("-" * 60)
    query3 = "python programming"
    
    print("\n🔍 Basic mode:")
    results5 = rag.handle_query(query3, mode="basic")
    print(f"   Got {len(results5)} results")
    
    print("\n🔍 Deep mode (different cache key):")
    results6 = rag.handle_query(query3, mode="deep")
    print(f"   Got {len(results6)} results")
    
    print("\n🔍 Basic mode again (should use cache):")
    results7 = rag.handle_query(query3, mode="basic")
    print(f"   Got {len(results7)} results")
    
    # Show cache statistics
    print("\n\n📊 Cache Statistics")
    print("-" * 60)
    stats = rag.query_cache.get_stats()
    print(f"Total cached queries: {stats['total_cached_queries']}")
    print("\nCached query+mode combinations:")
    for key in stats['cached_keys']:
        print(f"  - Query: '{key[0]}', Mode: '{key[1]}'")
    
    # Clear cache
    print("\n\n🧹 Clearing Cache")
    print("-" * 60)
    rag.query_cache.clear()
    
    print("\n🔍 Query after cache clear (will hit database again):")
    results8 = rag.handle_query(query1, mode="basic")
    print(f"   Got {len(results8)} results")
    
    # Close connection
    rag.close_neo4j()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


def simple_usage_example():
    """Simple example of typical usage"""
    
    rag = BuildingRag()
    rag.setup_neo4j_graph()
    
    # Basic search (top 3 results)
    results = rag.handle_query("machine learning", mode="basic")
    
    # Deep search (top 6 results with graph expansion)
    results = rag.handle_query("neural networks", mode="deep")
    
    # Same query again - uses cache automatically!
    results = rag.handle_query("neural networks", mode="deep")
    
    rag.close_neo4j()


if __name__ == "__main__":
    # Run the full demo
    demo_cache_usage()
    
    # Or use the simple example:
    # simple_usage_example()
