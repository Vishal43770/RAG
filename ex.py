# import os
# import pandas as pd
# import sqlite3


# class BuildingRag:
#     def __init__(self):
#         pass

#     def know_data(self):
#         os.makedirs("databases", exist_ok=True)
#         dbpath = "databases/VISHALLL.db"

#         conn = sqlite3.connect(dbpath)
#         cursor = conn.cursor()

#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         print("Tables:", cursor.fetchall())

#         cursor.execute("PRAGMA table_info(documents);")
#         print("Schema:", cursor.fetchall())

#         df = pd.read_sql_query("SELECT * FROM documents LIMIT 5;", conn)
#         print(df.head())

#         conn.close()

#     def data_merging(self):
#         db_files = [
#             "databases/VISHALLL.db",
#             "databases/SHIVA.db",
#             "databases/UDAY.db",
#             "databases/MASTER.db",
#         ]

#         def get_schema(db_path, table_name="documents"):
#             conn = sqlite3.connect(db_path)
#             cursor = conn.cursor()

#             cursor.execute(f"PRAGMA table_info({table_name});")
#             schema = cursor.fetchall()

#             conn.close()

#             # return (column_name, column_type)
#             return {(col[1], col[2]) for col in schema}

#         # Collect schemas
#         schemas = {}
#         for db in db_files:
#             schemas[db] = get_schema(db)

#         # Compare schemas
#         base_schema = list(schemas.values())[0]

#         for db, schema in schemas.items():
#             if schema != base_schema:
#                 print(f"❌ Schema mismatch in {db}")
#                 print("Difference:", schema.symmetric_difference(base_schema))
#                 raise Exception("Schemas are NOT compatible")

#         print("✅ All schemas match — safe to merge")


# if __name__ == "__main__":
#     BuildingRag().data_merging()












































import os
import pandas as pd
import sqlite3
import nltk


class SessionQueryCache:
    """In-memory cache for query results to avoid redundant searches during a session"""
    
    def __init__(self):
        """Initialize empty cache"""
        self.cache = {}  # { (query_text, mode): [chunks] }
    
    def get(self, query_text, mode):
        """
        Retrieve cached results for a query+mode combination
        
        Args:
            query_text: The search query
            mode: Search mode ('basic', 'fast', 'deep', 'web_search')
        
        Returns:
            Cached results if found, None otherwise
        """
        key = (query_text.strip().lower(), mode)
        return self.cache.get(key)
    
    def set(self, query_text, mode, results):
        """
        Store results in cache for a query+mode combination
        
        Args:
            query_text: The search query
            mode: Search mode ('basic', 'fast', 'deep', 'web_search')
            results: The search results to cache
        """
        key = (query_text.strip().lower(), mode)
        self.cache[key] = results
    
    def clear(self):
        """Clear all cached results"""
        self.cache.clear()
        print("🧹 Query cache cleared")
    
    def get_stats(self):
        """Get cache statistics"""
        return {
            'total_cached_queries': len(self.cache),
            'cached_keys': list(self.cache.keys())
        }


class BuildingRag:
    def __init__(self):
        nltk.download('punkt')
        self.query_cache = SessionQueryCache()
        print("✅ Query cache initialized")
        

    def know_data(self):
        os.makedirs("databases", exist_ok=True)
        dbpath = "databases/merged.db"
        conn = sqlite3.connect(dbpath)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print("Tables:", cursor.fetchall())
        cursor.execute("PRAGMA table_info(documents);")
        print("Schema:", cursor.fetchall())
        df = pd.read_sql_query("SELECT * FROM document_chunks LIMIT 20;", conn)
        print(df.head(20))
        conn.close()
    def data_merging(self):
        db_files = [
            "databases/VISHALLL.db",
            "databases/SHIVA.db",
            "databases/UDAY.db",
            "databases/MASTER.db",
        ]
        def get_schema(db_path, table_name="documents"):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = cursor.fetchall()
            conn.close()
            return {(col[1], col[2]) for col in schema}
        schemas = {}
        for db in db_files:
            schemas[db] = get_schema(db)
        base_schema = list(schemas.values())[0]
        for db, schema in schemas.items():
            if schema != base_schema:
                print(f"❌ Schema mismatch in {db}")
                print("Difference:", schema.symmetric_difference(base_schema))
                raise Exception("Schemas are NOT compatible")

        print("✅ All schemas match — safe to merge")

    def merge_databases(self, output_db="databases/merged.db", batch_size=200):
        db_files = [
            "databases/VISHALLL.db",
            "databases/SHIVA.db",
            "databases/UDAY.db",
            "databases/MASTER.db",  
        ]
        
        if os.path.exists(output_db):
            os.remove(output_db)
            print(f"🗑️  Removed existing {output_db}")
        
        first_conn = sqlite3.connect(db_files[0])
        first_cursor = first_conn.cursor()
        first_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='documents';")
        result = first_cursor.fetchone()
        
        if result is None:
            print(f"❌ Error: 'documents' table not found in {db_files[0]}")
            print("Available tables:")
            first_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = first_cursor.fetchall()
            for table in tables:
                print(f"  - {table[0]}")
            first_conn.close()
            raise ValueError(f"Table 'documents' does not exist in {db_files[0]}. Please check your database structure.")
        
        create_table_sql = result[0]
        first_conn.close()
        
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        
        total_inserted = 0
        for db_index, db in enumerate(db_files, 1):
            print(f"\n📂 [{db_index}/{len(db_files)}] Processing {db}...")
            
            source_conn = sqlite3.connect(db)
            source_cursor = source_conn.cursor()
            
            source_cursor.execute("SELECT COUNT(*) FROM documents;")
            total_rows = source_cursor.fetchone()[0]
            print(f"   Total rows: {total_rows:,}")
            
            offset = 0
            batch_num = 0
            while offset < total_rows:
                batch_num += 1
                source_cursor.execute(f"SELECT * FROM documents LIMIT {batch_size} OFFSET {offset};")
                rows = source_cursor.fetchall()
                
                if not rows:
                    break
                
                column_count = len(rows[0])
                placeholders = ','.join(['?' for _ in range(column_count)])
                
                cursor.executemany(f"INSERT INTO documents VALUES ({placeholders})", rows)
                conn.commit()
                offset += batch_size
                total_inserted += len(rows)
                progress = min(100, (offset / total_rows) * 100)
                print(f"   Batch {batch_num}: {len(rows)} rows | Progress: {progress:.1f}% ({offset:,}/{total_rows:,})", end='\r')            
            source_conn.close()
        conn.close()
    def chunk_documents(self):
        dbpath = "databases/merged.db"
        conn = sqlite3.connect(dbpath)
        cursor = conn.cursor()

        # cursor.execute("""CREATE TABLE document_chunks(id INTEGER NOT NULL,chunk_id INTEGER NOT NULL,chunk_text TEXT NOT NULL CHECK (length(chunk_text)<=2000),FOREIGN KEY (id) REFERENCES documents(id))""")

        cursor.execute(""" select id , cleaned_content from documents;
        """)
        conn.commit()
        documents=cursor.fetchall()
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter=RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        for dock_id, text in documents:
            if not text:
                continue
            chunkid=0

            chunks=splitter.split_text(text)
            for chunk_text in chunks:
                cursor.execute("""
                INSERT INTO document_chunks(id, chunk_id, chunk_text)
                VALUES (?, ?, ?);
                """, (dock_id, chunkid, chunk_text))
                chunkid+=1
        conn.commit()
        conn.close()
        
    def setup_neo4j_graph(self):
        """Initialize Neo4j connection and create constraints"""
        from neo4j import GraphDatabase
        from dotenv import load_dotenv
        
        load_dotenv()
        
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "ragpassword")
            )
        )
        print("🔗 Connected to Neo4j")
        # Create constraints and indexes
        with self.neo4j_driver.session() as session:
            # Unique constraint
            session.run("""
                CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
            print("✅ Created unique constraint on Chunk.id")
            
            # Vector index for embeddings
            session.run("""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            print("✅ Created vector index for embeddings")
    
    def ingest_to_neo4j(self, batch_size=100, limit=None):
        """Load chunks from SQLite to Neo4j with embeddings  bvh"""
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm        
        from dotenv import load_dotenv
        
        print("\n📊 Starting Neo4j ingestion...")
        
        # Load embedding model
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded")
        
        # Connect to SQLite
        conn = sqlite3.connect("databases/merged.db")
        cursor = conn.cursor()
        
        # Get total count
        query = "SELECT COUNT(*) FROM document_chunks"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        total = cursor.fetchone()[0]
        
        # Fetch chunks with document URLs
        query = """
            SELECT dc.rowid, dc.id, dc.chunk_id, dc.chunk_text, d.url
            FROM document_chunks dc
            LEFT JOIN documents d ON dc.id = d.id
        """
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        
        batch = []
        processed = 0
        
        for row in tqdm(cursor, total=total, desc="Ingesting chunks"):
            chunk_rowid, doc_id, chunk_id, text, url = row
            
            # Generate embedding
            embedding = embedding_model.encode(text).tolist()
            
            batch.append({
                'chunk_rowid': chunk_rowid,
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'text': text,
                'url': url or '',
                'embedding': embedding
            })
            
            if len(batch) >= batch_size:
                self._write_neo4j_batch(batch)
                processed += len(batch)
                batch = []
        if batch:
            self._write_neo4j_batch(batch)
            processed += len(batch)
        
        conn.close()
        print(f"\n✅ Ingested {processed:,} chunks to Neo4j")
    
    def _write_neo4j_batch(self, batch):
        """Write batch of chunks to Neo4j"""
        with self.neo4j_driver.session() as session:
            session.run("""
                UNWIND $batch AS row
                MERGE (d:Document {id: row.doc_id})
                SET d.url = row.url
                MERGE (c:Chunk {id: row.chunk_rowid})
                SET c.doc_id = row.doc_id,
                    c.chunk_id = row.chunk_id,
                    c.text = row.text,
                    c.url = row.url,
                    c.embedding = row.embedding
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, batch=batch)
    
    def create_similarity_edges(self, k=5, threshold=0.7):
        """Create SIMILAR_TO edges between semantically similar chunks"""
        print(f"\n🔗 Creating similarity edges (k={k}, threshold={threshold})...")
        
        with self.neo4j_driver.session() as session:
            # Get total chunk count
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            total = result.single()['count']
            print(f"Processing {total:,} chunks...")
            
            # Create similarity edges using vector index
            session.run("""
                MATCH (c1:Chunk)
                CALL db.index.vector.queryNodes('chunk_embeddings', $k + 1, c1.embedding)
                YIELD node AS c2, score
                WHERE c1 <> c2 AND score >= $threshold
                MERGE (c1)-[r:SIMILAR_TO]-(c2)
                SET r.score = score
            """, k=k, threshold=threshold)
            
            # Count created edges
            result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS count")
            edge_count = result.single()['count']
            print(f"✅ Created {edge_count:,} similarity edges")
    
    def vector_search(self, query_text, k=10):
        """Search using vector similarity"""
        from sentence_transformers import SentenceTransformer
        
        # Generate query embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text).tolist()
        
        with self.neo4j_driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                YIELD node, score
                RETURN node.text AS text, node.url AS url, node.id AS id, score
                ORDER BY score DESC
            """, k=k, query_embedding=query_embedding)
            
            results = []
            for record in result:
                results.append({
                    'text': record['text'],
                    'url': record['url'],
                    'id': record['id'],
                    'score': record['score']
                })
            
            return results
    
    def graph_search(self, query_text, k_seeds=3, depth=2):
        """Search using graph traversal from seed nodes"""
        from sentence_transformers import SentenceTransformer
        
        # Get seed nodes via vector search
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text).tolist()
        
        with self.neo4j_driver.session() as session:
            # Find seed chunks
            seed_result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                YIELD node
                RETURN collect(node.id) AS seed_ids
            """, k=k_seeds, query_embedding=query_embedding)
            
            seed_ids = seed_result.single()['seed_ids'] 
            
            # Expand via graph - using f-string for depth since Cypher doesn't allow parameterized variable-length paths
            result = session.run(f"""
                MATCH (seed:Chunk)
                WHERE seed.id IN $seed_ids
                MATCH path = (seed)-[:SIMILAR_TO*1..{depth}]-(related:Chunk)
                WITH DISTINCT related, length(path) AS distance
                RETURN related.text AS text, related.url AS url, related.id AS id, distance
                ORDER BY distance ASC
                LIMIT 20
            """, seed_ids=seed_ids)
            
            results = []
            for record in result:
                results.append({
                    'text': record['text'],
                    'url': record['url'],
                    'id': record['id'],
                    'distance': record['distance']
                })
            
            return results
    
    def handle_query(self, query, mode="basic"):
        """
        Main entry point for queries with caching support
        Args:
            query: Search query text
            mode: Search mode - 'basic', 'fast', 'deep', 'web_search'
        Returns:
            List of search results based on the mode
        """
        # 1. Check cache first
        cached = self.query_cache.get(query, mode)
        if cached:
            print(f"⚡ Using cached results for '{query}' [{mode}]")
            return cached
        # 2. Perform search based on mode
        print(f"🔍 Performing {mode} search for: '{query}'")
        if mode == "basic":
            # Vector-only search, top 3 results
            results = self.vector_search(query, k=3)
        
        elif mode == "fast":
            # Fast vector search, top 3 results
            results = self.vector_search(query, k=3)
        elif mode == "deep":
            # Hybrid search with graph expansion, top 6 results
            results = self.hybrid_search(query, k_vector=3, expand_depth=2)
        elif mode == "web_search":
            # External web search (placeholder - implement with your preferred API)
            print("⚠️  Web search mode not yet implemented")
            results = []
        
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'basic', 'fast', 'deep', or 'web_search'")
        
        # 3. Cache the results
        self.query_cache.set(query, mode, results)
        print(f"💾 Cached {len(results)} results for '{query}' [{mode}]")
        
        return results
    
    def hybrid_search(self, query_text, k_vector=5, expand_depth=1, decay_factor=0.85):
        """Hybrid search: vector seeds + graph expansion with decayed scoring"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text).tolist()
        
        with self.neo4j_driver.session() as session:
            result = session.run(f"""
                // Step 1: Vector search to get seed nodes with scores
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                YIELD node AS seed, score
                
                // Step 2: Expand through SIMILAR_TO graph edges
                OPTIONAL MATCH path = (seed)-[:SIMILAR_TO*1..{expand_depth}]-(related:Chunk)
                
                WITH 
                    seed, 
                    score AS seed_score,
                    related,
                    CASE 
                        WHEN related IS NULL THEN 0 
                        ELSE length(path) 
                    END AS hop_distance
                
                // Step 3: Get unique chunks with their minimum hop distance
                WITH DISTINCT 
                    COALESCE(related, seed) AS chunk,
                    MIN(hop_distance) AS hop_distance,
                    MAX(seed_score) AS seed_score  // max score if multiple paths reach same chunk
                
                RETURN 
                    chunk.text AS text,
                    chunk.url AS url,
                    chunk.id AS id,
                    seed_score,
                    hop_distance
                ORDER BY seed_score DESC, hop_distance ASC
                LIMIT 50
            """, k=k_vector, query_embedding=query_embedding)
            
            results = []
            seen_ids = set()
            
            for record in result:
                chunk_id = record['id']
                if chunk_id in seen_ids:
                    continue
                
                hop_distance = record['hop_distance'] or 0
                seed_score = record['seed_score'] or 0.0
                
                # Apply decay: deeper nodes get lower scores
                # hop_distance=0 means it's a seed node (no decay)
                # hop_distance=1 means 1 hop away (apply decay once)
                final_score = seed_score * (decay_factor ** hop_distance)
                
                results.append({
                    'text': record['text'],
                    'url': record['url'],
                    'id': chunk_id,
                    'score': round(final_score, 4),
                    'hop_distance': hop_distance
                })
                seen_ids.add(chunk_id)
            
            # Sort by final score descending
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
    


    
    def close_neo4j(self):
        """Close Neo4j connection"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
            print("🔒 Neo4j connection closed")
    
    def check_neo4j_data_exists(self):
        """Check if Neo4j already has data ingested"""
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            count = result.single()['count']
            return count > 0
    
    def get_neo4j_stats(self):
        """Get current Neo4j database statistics"""
        with self.neo4j_driver.session() as session:
            # Count chunks
            chunk_result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            chunk_count = chunk_result.single()['count']
            
            # Count documents
            doc_result = session.run("MATCH (d:Document) RETURN count(d) AS count")
            doc_count = doc_result.single()['count']
            
            # Count similarity edges
            edge_result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS count")
            edge_count = edge_result.single()['count']
            
            return {
                'chunks': chunk_count,
                'documents': doc_count,
                'similarity_edges': edge_count
            }
    
    def clear_neo4j_data(self):
        """Clear all data from Neo4j (use with caution!)"""
        print("⚠️  WARNING: This will delete ALL data from Neo4j!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("❌ Cancelled")

        
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️  All Neo4j data deleted")
    
    def run_full_pipeline(self, setup_all=True):
        """
        Orchestrate the full RAG pipeline from start to finish
        
        Args:
            setup_all: If True, runs ALL setup steps. If False, only connects to existing Neo4j data
        """
        print("\n" + "="*70)
        print("🚀 RAG SYSTEM FULL PIPELINE")
        print("="*70)
        
        if setup_all:
            print("\n[1/6] 📊 Checking data...")
            self.know_data()
            
            print("\n[2/6] 🔄 Merging databases...")
            self.merge_databases()
            
            print("\n[3/6] ✂️  Chunking documents...")
            self.chunk_documents()
            
            print("\n[4/6] 🔗 Setting up Neo4j...")
            self.setup_neo4j_graph()
            
            print("\n[5/6] 📥 Ingesting to Neo4j...")
            self.ingest_to_neo4j(batch_size=100)
            
            print("\n[6/6] 🕸️  Creating similarity edges...")
            self.create_similarity_edges(k=5, threshold=0.7)
        else:
            print("\n[1/1] 🔗 Connecting to Neo4j...")
            self.setup_neo4j_graph()
        
        # Show stats
        print("\n" + "="*70)
        print("📊 DATABASE STATISTICS")
        print("="*70)
        stats = self.get_neo4j_stats()
        print(f"  Chunks: {stats['chunks']:,}")
        print(f"  Documents: {stats['documents']:,}")
        print(f"  Similarity Edges: {stats['similarity_edges']:,}")
        
        print("\n✅ Pipeline complete! Ready for queries.")
        print("="*70 + "\n")
    
    def chat_with_rag(self, mode="basic", gemini_api_key=None):
        """
        Interactive chat with RAG using Neo4j vector search (no FAISS needed)
        
        Args:
            mode: Search mode - 'basic', 'fast', 'deep' (default: 'basic')
            gemini_api_key: Optional Gemini API key for LLM responses
        """
        print("\n" + "="*70)
        print(f"🤖 RAG CHAT INTERFACE")
        print("="*70)
        print("\n📋 Available Modes:")
        print("  • basic - Fast vector search (3 results)")
        print("  • fast  - Same as basic (3 results)")
        print("  • deep  - Hybrid search with graph (6 results, recommended)")
        print("\n⌨️  Commands:")
        print("  • Type your question to search")
        print("  • 'mode <basic|fast|deep>' to change search mode")
        print("  • 'stats' to see cache statistics")
        print("  • 'clear' to clear cache")
        print("  • 'exit' or 'quit' to end")
        print("="*70)
        print(f"\n🎯 Current Mode: {mode.upper()}")
        if gemini_api_key:
            print("🤖 LLM: Enabled (Gemini)")
        else:
            print("📝 LLM: Disabled (search results only)")
        print("="*70 + "\n")
        
        # Optional: Initialize LLM if API key provided
        llm = None
        if gemini_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                os.environ["GOOGLE_API_KEY"] = "[AIzaSyCKwZVmRqIlnzFvDyJx0YBfzEyG7WKNjp4]"
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
                print("✅ Gemini LLM initialized for enhanced responses\n")
            except Exception as e:
                print(f"⚠️  Could not initialize LLM: {e}")
                print("📝 Continuing with search-only mode\n")
        
        current_mode = mode
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in {"exit", "quit"}:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == "stats":
                    stats = self.query_cache.get_stats()
                    print(f"\n📊 Cache Statistics:")
                    print(f"  Total cached queries: {stats['total_cached_queries']}")
                    if stats['cached_keys']:
                        print("  Cached queries:")
                        for query, m in stats['cached_keys']:
                            print(f"    - '{query[:50]}...' [{m}]")
                    print()
                    continue
                
                elif user_input.lower() == "clear":
                    self.query_cache.clear()
                    continue
                
                elif user_input.lower().startswith("mode "):
                    new_mode = user_input.split()[1].lower()
                    if new_mode in ["basic", "fast", "deep"]:
                        current_mode = new_mode
                        print(f"✅ Mode changed to: {current_mode.upper()}\n")
                    else:
                        print("❌ Invalid mode. Use: basic, fast, or deep\n")
                    continue
                
                # Perform search
                print(f"\n🔍 Searching [{current_mode}]...")
                results = self.handle_query(user_input, mode=current_mode)
                
                if not results:
                    print("❌ No results found.\n")
                    continue
                
                # Display results
                print(f"\n📚 Search Results ({len(results)} chunks found)")
                print("=" * 70)
                
                # Show ranked results with URLs
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    print(f"\n🔹 Rank #{i} | Score: {score:.4f}" if isinstance(score, (int, float)) else f"\n🔹 Rank #{i} | Score: {score}")
                    
                    if result.get('url'):
                        print(f"   🔗 Source: {result['url']}")
                    
                    text = result.get('text', '')
                    # Show snippet
                    if len(text) > 300:
                        print(f"   📄 {text[:300]}...")
                    else:
                        print(f"   📄 {text}")
                    
                    if i < len(results):
                        print("   " + "-" * 66)
                
                print("\n" + "=" * 70)
                
                # Optional LLM response
                if llm:
                    try:
                        print("\n🤖 Generating AI Response...")
                        print("=" * 70)
                        
                        # Use top 3 chunks for context
                        top_chunks = results[:3]
                        context_parts = []
                        
                        for i, chunk in enumerate(top_chunks, 1):
                            text = chunk.get('text', '')
                            url = chunk.get('url', 'N/A')
                            context_parts.append(f"[Source {i} - {url}]\n{text}")
                        
                        context = "\n\n".join(context_parts)
                        
                        prompt = f"""You are a helpful AI assistant. Based on the following context from a knowledge base, provide a comprehensive answer to the user's question.

Context:
{context}

Question: {user_input}

Provide a detailed, accurate answer based on the context above. If the context doesn't contain enough information, say so.

Answer:"""
                        
                        # Stream the response token by token
                        print("\n", end="", flush=True)
                        for chunk in llm.stream(prompt):
                            print(chunk.content, end="", flush=True)
                        print("\n")
                        print("=" * 70)
                        
                        # Show source links ranked by score
                        print("\n📌 Sources Used (Ranked by Relevance):")
                        for i, chunk in enumerate(top_chunks, 1):
                            if chunk.get('url'):
                                score = chunk.get('score', 'N/A')
                                if isinstance(score, (int, float)):
                                    print(f"  [{i}] {chunk['url']} (Score: {score:.4f})")
                                else:
                                    print(f"  [{i}] {chunk['url']}")
                        print()
                        
                    except Exception as e:
                        print(f"\n⚠️  LLM Error: {e}\n")
                        print("📝 Showing search results only.\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")

def main():
    """
    Main entry point - Run full RAG pipeline and start chat
    """
    import sys
    
    rag = BuildingRag()
    
    print("\n🎯 What would you like to do?")
    print("  1. Run FULL pipeline (merge, chunk, ingest, then chat)")
    print("  2. Connect to EXISTING Neo4j data and chat")
    print("  3. Just chat (assumes Neo4j is already set up) [DEFAULT]")
    
    choice = input("\nEnter choice (1/2/3) [default: 3]: ").strip() or "3"
    
    if choice == "1":
        rag.run_full_pipeline(setup_all=True)
    elif choice == "2":
        rag.run_full_pipeline(setup_all=False)
    elif choice == "3":
        rag.setup_neo4j_graph()
    else:
        print("❌ Invalid choice. Exiting.")
        sys.exit(1)
    
    # Ask for Gemini API key (optional)
    use_llm = input("\nUse Gemini LLM for enhanced responses? (y/n): ").strip().lower()
    gemini_key = None
    if use_llm == 'y':
        gemini_key = input("Enter Gemini API key: ").strip()
    
    # Start chat
    rag.chat_with_rag(mode="deep", gemini_api_key=gemini_key)
    
    # Cleanup
    rag.close_neo4j()


if __name__ == "__main__":
    main()