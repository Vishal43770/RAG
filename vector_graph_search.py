"""
Vector-Seeded Graph Search for RAG
===================================

This module implements a hybrid search strategy:
1. Vector Search → Find semantically relevant seed nodes (entry points)
2. Graph Traversal → Expand context via SIMILAR_TO relationships

Why This Approach?
------------------
✅ Vector search finds the most relevant starting points
✅ Graph traversal discovers related context through relationships
✅ Combines semantic similarity with structural relationships
✅ Multi-hop exploration uncovers non-obvious connections
"""

import os
import sqlite3
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()


class VectorGraphSearch:
    """
    Semantic search using vector embeddings + graph relationships
    """
    
    def __init__(self):
        """Initialize Neo4j connection and embedding model"""
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "ragpassword")
            )
        )
        print("✅ Connected to Neo4j")
        
        # Load embedding model (same as used during ingestion)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded")
    
    def vector_seed_search(
        self, 
        query: str, 
        k_seeds: int = 3,
        depth: int = 2,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid vector-seeded graph search
        
        Steps:
        1. Encode query → vector
        2. Find k_seeds most similar chunks (vector search)
        3. Expand via graph (SIMILAR_TO edges) up to `depth` hops
        4. Return ranked results with context
        
        Args:
            query: User's search query
            k_seeds: Number of seed nodes to start from
            depth: How many hops to traverse (1-3 recommended)
            min_similarity: Minimum similarity score for SIMILAR_TO edges
            
        Returns:
            List of results with text, URL, scores
        """
        print(f"\n🔍 Query: {query}")
        print(f"📊 Config: {k_seeds} seeds, depth={depth}, min_sim={min_similarity}")
        
        # Step 1: Encode query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        with self.driver.session() as session:
            # Step 2: Find seed nodes via vector search
            print(f"\n1️⃣ Finding {k_seeds} seed nodes via vector search...")
            seed_result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                YIELD node, score
                RETURN 
                    node.id AS chunk_id,
                    node.text AS text,
                    node.url AS url,
                    score
                ORDER BY score DESC
            """, k=k_seeds, query_embedding=query_embedding)
            
            seeds = []
            for record in seed_result:
                seeds.append({
                    'chunk_id': record['chunk_id'],
                    'text': record['text'],
                    'url': record['url'],
                    'vector_score': record['score'],
                    'type': 'seed'
                })
            
            print(f"   Found {len(seeds)} seed chunks")
            
            if not seeds:
                print("⚠️ No seed nodes found")
                return []
            
            # Step 3: Expand via graph traversal
            print(f"\n2️⃣ Expanding via graph ({depth}-hop traversal)...")
            seed_ids = [s['chunk_id'] for s in seeds]
            
            # Graph expansion query
            expansion_result = session.run(f"""
                // Start from seed nodes
                MATCH (seed:Chunk)
                WHERE seed.id IN $seed_ids
                
                // Traverse SIMILAR_TO relationships
                OPTIONAL MATCH path = (seed)-[r:SIMILAR_TO*1..{depth}]-(related:Chunk)
                WHERE ALL(rel IN relationships(path) WHERE rel.score >= $min_similarity)
                
                // Collect all unique chunks (seeds + related)
                WITH seed, related, 
                     CASE WHEN related IS NOT NULL 
                          THEN length(path) 
                          ELSE 0 
                     END AS hop_distance
                
                // Return distinct chunks
                WITH DISTINCT COALESCE(related, seed) AS chunk, 
                     MIN(hop_distance) AS distance
                
                RETURN 
                    chunk.id AS chunk_id,
                    chunk.text AS text,
                    chunk.url AS url,
                    distance
                ORDER BY distance ASC
                LIMIT 20
            """, seed_ids=seed_ids, min_similarity=min_similarity)
            
            # Combine results
            results = []
            seed_map = {s['chunk_id']: s for s in seeds}
            
            for record in expansion_result:
                chunk_id = record['chunk_id']
                distance = record['distance']
                
                # If it's a seed, include vector score
                if chunk_id in seed_map:
                    result = seed_map[chunk_id].copy()
                    result['distance'] = distance
                else:
                    result = {
                        'chunk_id': chunk_id,
                        'text': record['text'],
                        'url': record['url'],
                        'distance': distance,
                        'vector_score': None,
                        'type': 'expanded'
                    }
                
                results.append(result)
            
            print(f"   Expanded to {len(results)} total chunks")
            return results
    
    def display_results(self, results: List[Dict[str, Any]]) -> None:
        """Pretty-print search results"""
        print(f"\n{'='*70}")
        print(f"📋 SEARCH RESULTS ({len(results)} chunks)")
        print(f"{'='*70}\n")
        
        for i, result in enumerate(results, 1):
            result_type = result.get('type', 'unknown')
            distance = result.get('distance', '?')
            vector_score = result.get('vector_score')
            
            # Header
            if result_type == 'seed':
                print(f"🌟 {i}. SEED NODE (distance: {distance})")
                print(f"   Vector Score: {vector_score:.4f}")
            else:
                print(f"🔗 {i}. EXPANDED NODE (distance: {distance} hops)")
            
            # Content
            print(f"   URL: {result.get('url', 'N/A')}")
            print(f"   Text: {result['text'][:150]}...")
            print()
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("🔒 Neo4j connection closed")


# ============================================================================
# Example Usage
# ============================================================================

def example_basic_search():
    """Basic vector-seeded graph search"""
    search = VectorGraphSearch()
    
    query = input("\nEnter your question: ")
    
    # Search with 3 seed nodes, 2-hop expansion
    results = search.vector_seed_search(
        query=query,
        k_seeds=3,
        depth=2,
        min_similarity=0.7
    )
    
    # Display results
    search.display_results(results)
    search.close()


def example_deep_exploration():
    """Deeper graph exploration"""
    search = VectorGraphSearch()
    query = input("\nEnter your question: ")
    # Search with more seeds and deeper traversal
    results = search.vector_seed_search(
        query=query,
        k_seeds=5,
        depth=3,
        min_similarity=0.6  # Lower threshold = more connections
    )
    search.display_results(results)
    search.close()


# def example_web_search():
#     """Web search with vector-seeded graph expansion"""
#     search = VectorGraphSearch()
    
#     query = input("\nEnter your question: ")
    
#     # Perform web search to get initial chunks
#     print("\n1️⃣ Performing web search...")
#     web_results = search.web_search(query, top_k=5)
    
#     if not web_results:
#         print("⚠️ No web results found")
#         search.close()
#         return
    
#     print(f"   Found {len(web_results)} web results")
    
#     # Extract chunk IDs from web search results
#     chunk_ids = [r['chunk_id'] for r in web_results]
    
#     # Expand via graph traversal
#     print(f"\n2️⃣ Expanding via graph (2-hop traversal)...")
#     graph_results = search.vector_seed_search(
#         query=query,  # Query is used for vector similarity
#         seed_chunk_ids=chunk_ids,  # Use web search results as seeds
#         depth=2,
#         min_similarity=0.7
#     )
    
#     # Combine results
#     combined_results = []
    
#     # Add web results first
#     for result in web_results:
#         combined_results.append({
#             'chunk_id': result['chunk_id'],
#             'text': result['text'],
#             'url': result['url'],
#             'vector_score': result.get('vector_score'),
#             'type': 'web_seed'
#         })
    
#     # Add graph expansion results (excluding web seeds)
#     web_ids = set(chunk_ids)
#     for result in graph_results:
#         if result['chunk_id'] not in web_ids:
#             combined_results.append(result)
    
#     # Display results
#     search.display_results(combined_results)
#     search.close()
if __name__ == "__main__":
    import sys  
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "basic":
            example_basic_search() # for llm pass top 3 results
        elif mode == "fast": #classic vector search
            example_fast_search() # for llm pass top 3 results
        elif mode == "deep":
            example_deep_exploration()  # for llm pass top 6 results
        elif mode == "web_search":
            example_web_search() # appliy question and chunks perfoem semeralty scroe pass only top 3 results
        else:
            print("Unknown mode. Use: basic, deep, or web_search")
    else:
        # Default: basic search
        print("\n" + "="*70)
        print("  VECTOR-SEEDED GRAPH SEARCH")
        print("="*70)
        print("\nModes:")
        print("  python vector_graph_search.py basic    # Standard search (3 seeds, 2 hops)")
        print("  python vector_graph_search.py deep     # Deep exploration (5 seeds, 3 hops)")
        print("\nRunning basic mode...\n")
        example_basic_search()
