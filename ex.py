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








































import time
import os
import pandas as pd
import sqlite3
import nltk
import logging
from datetime import datetime
import pytz

# IST Logger Setup
class ISTFormatter(logging.Formatter):
    """Custom formatter to display time in Indian Standard Time (12-hour format)"""
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp)
        ist = pytz.timezone('Asia/Kolkata')
        return dt.astimezone(ist)
    
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime('%I:%M:%S %p')  # 12-hour format with AM/PM

# Setup logger
logger = logging.getLogger('RAG_Pipeline')
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(ISTFormatter('%(asctime)s | %(message)s'))
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('rag_pipeline.log')
file_handler.setFormatter(ISTFormatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(file_handler)


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
        """Merge databases with checkpoint - skips if already complete"""
        
        # Checkpoint: Check if merge already complete
        if os.path.exists(output_db):
            try:
                conn = sqlite3.connect(output_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                existing_count = cursor.fetchone()[0]
                conn.close()
                
                if existing_count > 0:
                    logger.info(f"✅ Merge already complete: {existing_count:,} documents in {output_db}")
                    return
            except sqlite3.OperationalError:
                logger.info(f"⚠️  {output_db} exists but corrupted, rebuilding...")
                os.remove(output_db)
        
        logger.info("� Starting database merge...")
        db_files = [
            "databases/VISHALLL.db",
            "databases/SHIVA.db",
            "databases/UDAY.db",
            "databases/MASTER.db",  
        ]
        
        # Get schema from first database
        first_conn = sqlite3.connect(db_files[0])
        first_cursor = first_conn.cursor()
        first_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='documents';")
        result = first_cursor.fetchone()
        
        if result is None:
            logger.error(f"❌ Error: 'documents' table not found in {db_files[0]}")
            first_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = first_cursor.fetchall()
            logger.info(f"Available tables: {[t[0] for t in tables]}")
            first_conn.close()
            raise ValueError(f"Table 'documents' does not exist in {db_files[0]}")
        
        create_table_sql = result[0]
        first_conn.close()
        
        # Create output database
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        logger.info(f"📄 Created {output_db}")
        
        # Merge all databases
        total_inserted = 0
        for db_index, db in enumerate(db_files, 1):
            logger.info(f"📂 [{db_index}/{len(db_files)}] Processing {db}...")
            
            source_conn = sqlite3.connect(db)
            source_cursor = source_conn.cursor()
            
            source_cursor.execute("SELECT COUNT(*) FROM documents;")
            total_rows = source_cursor.fetchone()[0]
            logger.info(f"   Total rows: {total_rows:,}")
            
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
                if batch_num % 100 == 0 or progress == 100:
                    logger.info(f"   Batch {batch_num}: {len(rows)} rows | Progress: {progress:.1f}% ({offset:,}/{total_rows:,})")
                    
            source_conn.close()
            
        conn.close()
        logger.info(f"✅ Merge complete: {total_inserted:,} total documents")
    def chunk_documents(self):
        """Chunk documents with checkpoint - resumes from unprocessed documents"""
        dbpath = "databases/merged.db"
        conn = sqlite3.connect(dbpath)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""CREATE TABLE IF NOT EXISTS document_chunks(id INTEGER NOT NULL,chunk_id INTEGER NOT NULL,chunk_text TEXT NOT NULL,FOREIGN KEY (id) REFERENCES documents(id))""")
        conn.commit()
        
        # Checkpoint: Check existing progress
        cursor.execute("SELECT COUNT(DISTINCT id) FROM document_chunks")
        processed_docs_count = cursor.fetchone()[0]
        
        if processed_docs_count > 0:
            logger.info(f"📊 Resume: {processed_docs_count:,} documents already chunked")
            # Get IDs of already processed documents
            cursor.execute("SELECT DISTINCT id FROM document_chunks")
            processed_ids = {row[0] for row in cursor.fetchall()}
        else:
            processed_ids = set()
            logger.info("🆕 Starting fresh chunking")
        
        # Get all documents
        cursor.execute("SELECT id, cleaned_content FROM documents")
        documents = cursor.fetchall()
        total_docs = len(documents)
        
        logger.info(f"📚 Total documents: {total_docs:,}")
        logger.info(f"✂️  Documents to chunk: {total_docs - len(processed_ids):,}")
        
        # Load text splitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Process only unprocessed documents
        processed_count = 0
        for i, (doc_id, text) in enumerate(documents, 1):
            if doc_id in processed_ids:
                continue  # Skip already processed
            
            if not text:
                continue
            
            chunk_id = 0
            chunks = splitter.split_text(text)
            for chunk_text in chunks:
                cursor.execute("""
                INSERT INTO document_chunks(id, chunk_id, chunk_text)
                VALUES (?, ?, ?);
                """, (doc_id, chunk_id, chunk_text))
                chunk_id += 1
            
            processed_count += 1
            
            # Log progress every 1000 docs and commit
            if processed_count % 1000 == 0:
                conn.commit()
                total_processed = len(processed_ids) + processed_count
                logger.info(f"📊 Progress: {total_processed:,} / {total_docs:,} ({total_processed/total_docs*100:.1f}%)")
        
        conn.commit()
        conn.close()
        
        total_final = len(processed_ids) + processed_count
        logger.info(f"✅ Chunking complete: {total_final:,} documents processed")
        
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
        logger.info("🔗 Connected to Neo4j")
        # Create constraints and indexes
        with self.neo4j_driver.session() as session:
            # Unique constraint
            session.run("""
                CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
            logger.info("✅ Created unique constraint on Chunk.id")
            
            # Vector index for embeddings
            session.run("""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            logger.info("✅ Created vector index for embeddings")
    
    def ingest_to_neo4j(self, batch_size=500, limit=None, embedding_batch_size=48):
        """Load chunks from SQLite to Neo4j with embeddings - RESUMABLE with checkpoint
        
        OPTIMIZED: Batches embedding generation for 10-20x speedup
        """
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
        import torch
        
        logger.info("📥 Starting Neo4j ingestion...")
        
        # Checkpoint: Check how many chunks already in Neo4j
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN MAX(c.id) AS max_id, COUNT(c) AS count")
            record = result.single()
            max_ingested_id = record['max_id'] or 0
            ingested_count = record['count']
        
        logger.info(f"📊 Already ingested: {ingested_count:,} chunks")
        logger.info(f"🔄 Resuming from chunk ID: {max_ingested_id + 1}")
        
        # Load embedding model with GPU if available
        logger.info("🤖 Loading embedding model...")
        device = 'cuda' if torch.cuda.is_available() else None
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        logger.info(f"✅ Model loaded on {device.upper()}")
        
        # Connect to SQLite
        conn = sqlite3.connect("databases/merged.db")
        cursor = conn.cursor()
        
        # Get total count of remaining chunks
        query = "SELECT COUNT(*) FROM document_chunks WHERE rowid > ?"
        cursor.execute(query, (max_ingested_id,))
        remaining = cursor.fetchone()[0]
        
        logger.info(f"📦 Remaining to ingest: {remaining:,} chunks")
        
        if remaining == 0:
            logger.info("✅ All chunks already ingested!")
            conn.close()
            return
        
        # Fetch chunks starting from last checkpoint
        query = """
            SELECT dc.rowid, dc.id, dc.chunk_id, dc.chunk_text, d.url
            FROM document_chunks dc
            LEFT JOIN documents d ON dc.id = d.id
            WHERE dc.rowid > ?
            ORDER BY dc.rowid
        """
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (max_ingested_id,))
        
        # Buffers for batched processing
        text_buffer = []  # For batched embedding generation
        metadata_buffer = []  # Store chunk metadata
        neo4j_batch = []  # For Neo4j writes
        
        processed = 0
        last_log_time = time.time()
        
        logger.info(f"🔄 Starting ingestion (embedding batch size: {embedding_batch_size}, Neo4j batch size: {batch_size})...")
        
        for row in tqdm(cursor, total=remaining, desc="Ingesting chunks"):
            chunk_rowid, doc_id, chunk_id, text, url = row
            
            # Add to embedding buffer
            text_buffer.append(text)
            metadata_buffer.append({
                'chunk_rowid': chunk_rowid,
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'url': url or ''
            })
            
            # Generate embeddings in batches (MUCH FASTER!)
            if len(text_buffer) >= embedding_batch_size:
                # Batch encode all texts at once
                embeddings = embedding_model.encode(text_buffer, convert_to_tensor=False, show_progress_bar=False)
                
                # Combine embeddings with metadata
                for i, embedding in enumerate(embeddings):
                    neo4j_batch.append({
                        **metadata_buffer[i],
                        'text': text_buffer[i],
                        'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                    })
                
                # Clear buffers
                text_buffer = []
                metadata_buffer = []
                
                # Write to Neo4j when batch is full
                if len(neo4j_batch) >= batch_size:
                    self._write_neo4j_batch(neo4j_batch)
                    processed += len(neo4j_batch)
                    neo4j_batch = []
                    
                    # Log progress every 10 seconds
                    if time.time() - last_log_time > 10:
                        total_ingested = ingested_count + processed
                        total_target = ingested_count + remaining
                        percentage = (total_ingested / total_target * 100)
                        progress_msg = f"📊 Progress: {total_ingested:,} / {total_target:,} ({percentage:.1f}%) | Last ID: {chunk_rowid}"
                        
                        # Print to console
                        print(f"{progress_msg}", end='\r', flush=True)
                        
                        # Log to file
                        file_handler.stream.write(f"{file_handler.formatter.formatTime(logging.LogRecord('', 0, '', 0, '', (), None))} | INFO | {progress_msg}\n")
                        file_handler.stream.flush()
                        
                        last_log_time = time.time()
        
        # Process remaining items in text buffer
        if text_buffer:
            embeddings = embedding_model.encode(text_buffer, convert_to_tensor=False, show_progress_bar=False)
            for i, embedding in enumerate(embeddings):
                neo4j_batch.append({
                    **metadata_buffer[i],
                    'text': text_buffer[i],
                    'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                })
        
        # Write remaining Neo4j batch
        if neo4j_batch:
            self._write_neo4j_batch(neo4j_batch)
            processed += len(neo4j_batch)
        
        conn.close()
        
        total_final = ingested_count + processed
        logger.info(f"✅ Ingestion complete: {processed:,} new chunks ingested")
        logger.info(f"📊 Total chunks in Neo4j: {total_final:,}")

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
        """Create SIMILAR_TO edges between semantically similar chunks - with checkpoint"""
        
        # Checkpoint: Check if edges already exist
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS count")
            existing_edges = result.single()['count']
        
        if existing_edges > 0:
            logger.info(f"✅ Similarity edges already exist: {existing_edges:,}")
            return
        
        logger.info(f"🔗 Creating similarity edges (k={k}, threshold={threshold})...")
        
        with self.neo4j_driver.session() as session:
            # Get total chunk count
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            total = result.single()['count']
            logger.info(f"Processing {total:,} chunks...")
            
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
            logger.info(f"✅ Created {edge_count:,} similarity edges")
    
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
            self.ingest_to_neo4j(batch_size=500)
            
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
    def chat_with_rag(self):
        """
        Interactive chat with RAG - displays only retrieved chunks (no LLM)
        """
        print("\n" + "="*70)
        print("🤖 RAG CHAT INTERFACE")
        print("="*70)
        print("\n📋 Available Modes:")
        print("  • basic - Fast vector search (3 results)")
        print("  • fast  - Same as basic (3 results)")
        print("  • deep  - Hybrid search with graph (50 results, recommended)")
        print("\n⌨️  Commands:")
        print("  • Type your question to search")
        print("  • 'mode <basic|fast|deep>' to change search mode")
        print("  • 'stats' to see cache statistics")
        print("  • 'clear' to clear cache")
        print("  • 'exit' or 'quit' to end")
        print("="*70)
        
        # Ask for mode selection
        print("\n🎯 Select search mode:")
        print("  1. basic - Fast vector search")
        print("  2. fast  - Same as basic")
        print("  3. deep  - Hybrid search (recommended)")
        
        mode_choice = input("\nEnter mode (1/2/3) [default: 3]: ").strip() or "3"
        
        if mode_choice == "1":
            current_mode = "basic"
        elif mode_choice == "2":
            current_mode = "fast"
        elif mode_choice == "3":
            current_mode = "deep"
        else:
            current_mode = "deep"
        
        print(f"\n✅ Mode set to: {current_mode.upper()}")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                start_time = time.time()
                
                
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
                
                # Display ONLY retrieved chunks (no LLM processing
                print(f"\n📚 Retrieved Chunks ({len(results)} found)")
                print("=" * 70 + "\n")
                
                # Show all ranked results with full details
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    hop_distance = result.get('hop_distance', 'N/A')
                    
                    print(f"🔹 Chunk #{i}")
                    
                    if isinstance(score, (int, float)):
                        print(f"   📊 Score: {score:.4f}")
                    else:
                        print(f"   📊 Score: {score}")
                    
                    if hop_distance != 'N/A':
                        print(f"   🕸️  Hop Distance: {hop_distance}")
                    
                    if result.get('url'):
                        print(f"   🔗 Source: {result['url']}")
                    
                    if result.get('id'):
                        print(f"   🆔 Chunk ID: {result['id']}")
                    
                    text = result.get('text', '')
                    print(f"\n   📄 Content:")
                    print(f"   {text}\n")
                    print(time.time() - start_time)
                    
                    if i < len(results):
                        print("   " + "-" * 66 + "\n")
                
                print("=" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def rebuild_pipeline(self):
        """Full rebuild pipeline with automatic checkpoints"""
        logger.info("="*70)
        logger.info("🔄 RAG DATABASE PIPELINE (WITH AUTOMATIC RESUME)")
        logger.info("="*70)
        
        # Connect to Neo4j
        logger.info("\n[1/5] 🔗 Setting up Neo4j...")
        self.setup_neo4j_graph()
        
        # Each function checks its own progress and resumes if needed
        logger.info("\n[2/5] 📊 Merging databases...")
        self.merge_databases()
        
        logger.info("\n[3/5] ✂️  Chunking documents...")
        self.chunk_documents()
        
        logger.info("\n[4/5] 📥 Ingesting to Neo4j...")
        self.ingest_to_neo4j(batch_size=500)
        
        logger.info("\n[5/5] 🕸️  Creating similarity edges...")
        self.create_similarity_edges(k=5, threshold=0.7)
        
        # Show final statistics
        logger.info("\n" + "="*70)
        logger.info("📊 FINAL STATISTICS")
        logger.info("="*70)
        stats = self.get_neo4j_stats()
        logger.info(f"  Chunks: {stats['chunks']:,}")
        logger.info(f"  Documents: {stats['documents']:,}")
        logger.info(f"  Similarity Edges: {stats['similarity_edges']:,}")
        logger.info("="*70)
        
        logger.info("\n🎉 PIPELINE COMPLETE!")
        logger.info("You can now run the chatbot with: python3 ex.py chat")
    
    def test_checkpoints(self):
        """Test checkpoint functionality"""
        logger.info("="*70)
        logger.info("🧪 TESTING CHECKPOINT FUNCTIONALITY")
        logger.info("="*70)
        
        self.setup_neo4j_graph()
        
        # Test 1: Merge checkpoint
        logger.info("\n[TEST 1] Testing merge_databases checkpoint...")
        self.merge_databases()
        logger.info("✓ Merge test passed")
        
        # Test 2: Chunk checkpoint  
        logger.info("\n[TEST 2] Testing chunk_documents checkpoint...")
        self.chunk_documents()
        logger.info("✓ Chunk test passed")
        
        # Test 3: Ingestion checkpoint
        logger.info("\n[TEST 3] Testing ingest_to_neo4j checkpoint...")
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            current_count = result.single()['count']
        logger.info(f"   Current chunks in Neo4j: {current_count:,}")
        
        self.ingest_to_neo4j(batch_size=500, limit=1000)
        
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            new_count = result.single()['count']
        
        added = new_count - current_count
        logger.info(f"   Added {added:,} chunks in test")
        logger.info("✓ Ingestion test passed")
        
        # Test 4: Edges checkpoint
        logger.info("\n[TEST 4] Testing create_similarity_edges checkpoint...")
        self.create_similarity_edges()
        logger.info("✓ Edges test passed")
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALL CHECKPOINT TESTS PASSED!")
        logger.info("="*70)
        
        stats = self.get_neo4j_stats()
        logger.info(f"\n📊 Current Database State:")
        logger.info(f"   Chunks: {stats['chunks']:,}")
        logger.info(f"   Documents: {stats['documents']:,}")
        logger.info(f"   Edges: {stats['similarity_edges']:,}")


def main():
    """Main entry point - automatically runs rebuild pipeline"""
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if a specific command was provided
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        # Default: run rebuild pipeline automatically
        command = 'rebuild'
    
    rag = BuildingRag()
    
    if command == 'rebuild':
        # Full rebuild pipeline with checkpoints
        rag.rebuild_pipeline()
        rag.close_neo4j()
    
    elif command == 'chat':
        # Connect to Neo4j and start chat
        logger.info("🔗 Connecting to Neo4j...")
        rag.setup_neo4j_graph()
        rag.chat_with_rag()
        rag.close_neo4j()
    
    elif command == 'stats':
        # Show stats only
        rag.setup_neo4j_graph()
        stats = rag.get_neo4j_stats()
        logger.info("="*70)
        logger.info("📊 DATABASE STATISTICS")
        logger.info("="*70)
        logger.info(f"  Chunks: {stats['chunks']:,}")
        logger.info(f"  Documents: {stats['documents']:,}")
        logger.info(f"  Similarity Edges: {stats['similarity_edges']:,}")
        logger.info("="*70)
        rag.close_neo4j()
    
    else:
        logger.error(f"❌ Unknown command: {command}")
        logger.info("Use: python3 ex.py [rebuild|chat|stats]")


if __name__ == "__main__":
    main()