"""
Neo4j Graph RAG Usage Examples

This demonstrates how to use the GraphRAG system implemented in ex.py
"""

from ex import BuildingRag

def example_setup():
    """Setup Neo4j and ingest data"""
    rag = BuildingRag()
    
    # 1. Connect to Neo4j and create indexes
    print("=== Setting up Neo4j ===")
    rag.setup_neo4j_graph()
    
    # 2. Ingest chunks (start with small sample for testing)
    print("\n=== Ingesting Data ===")
    rag.ingest_to_neo4j(batch_size=100, limit=1000)  # Start with 1000 chunks
    
    # 3. Create similarity edges
    print("\n=== Creating Graph Edges ===")
    rag.create_similarity_edges(k=5, threshold=0.7)
    
    rag.close_neo4j()
    print("\n✅ Setup complete!")

def example_vector_search():
    """Vector search example"""
    rag = BuildingRag()
    rag.setup_neo4j_graph()
    
    print("\n=== Vector Search ===")
    query =input("Enter your question: ") 
    results = rag.vector_search(query, k=5)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   URL: {result.get('url', 'N/A')}")
        print(f"   Text: {result['text']}")
    
    rag.close_neo4j()

def example_graph_search():
    """Graph traversal search example"""
    rag = BuildingRag()
    rag.setup_neo4j_graph()
    
    print("\n=== Graph Search ===")
    query =input("Enter your question: ") 
    print("QUESTION:",query)
    results = rag.graph_search(query, k_seeds=4, depth=1)
    starting
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Distance: {result['distance']}")
        print(f"   URL: {result.get('url', 'N/A')}")
        print(f"   Text: {result['text']}")
    
    rag.close_neo4j()

def example_hybrid_search():
    """Hybrid search example (vector + graph)"""
    rag = BuildingRag()
    rag.setup_neo4j_graph()
    
    print("\n=== Hybrid Search ===")
    query =input("Enter your question: ") 
    results = rag.hybrid_search(query, k_vector=5, expand_depth=1)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   URL: {result.get('url', 'N/A')}")
        print(f"   Text: {result['text']}")
    
    rag.close_neo4j()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python examples_neo4j.py setup       # Initial setup and ingestion")
        print("  python examples_neo4j.py vector      # Vector search demo")
        print("  python examples_neo4j.py graph       # Graph search demo")
        print("  python examples_neo4j.py hybrid      # Hybrid search demo")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        example_setup()
    elif command == "vector":
        example_vector_search()
    elif command == "graph":
        example_graph_search()
    elif command == "hybrid":
        example_hybrid_search()
    else:
        print(f"Unknown command: {command}")
