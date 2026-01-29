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


class BuildingRag:
    def __init__(self):
        pass

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
        create_table_sql = first_cursor.fetchone()[0]
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
        # cursor.execute("""
        #     CREATE TABLE document_chunks(
        #         id INTEGER NOT NULL,
        #         chunk_id INTEGER NOT NULL,
        #         chunk_text TEXT NOT NULL CHECK (length(chunk_text)<=2000),
        #         FOREIGN KEY (id) REFERENCES documents(id)
        #     )
        # """)
        cursor.execute(""" select id , cleaned_content from documents;
        """)
        documents=cursor.fetchall()
        for dock_id, text in documents:
            if not text:
                continue
            chunkid=0
            for i in range(0, len(text), 2000):
                chunk_text=text[i:i+2000]
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
        """Load chunks from SQLite to Neo4j with embeddings"""
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
        
        # Write remaining batch
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
    
    def hybrid_search(self, query_text, k_vector=5, expand_depth=1):
        """Combine vector search + graph expansion"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text).tolist()
        
        with self.neo4j_driver.session() as session:
            # Using f-string for depth since Cypher doesn't allow parameterized variable-length paths
            result = session.run(f"""
                // Vector search for initial nodes
                CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
                YIELD node AS seed, score
                
                // Expand via graph
                OPTIONAL MATCH (seed)-[:SIMILAR_TO*1..{expand_depth}]-(related:Chunk)
                
                // Return unique chunks with scores
                WITH seed, score, collect(DISTINCT related) AS neighbors
                UNWIND neighbors + [seed] AS chunk
                
                RETURN DISTINCT chunk.text AS text,
                       chunk.url AS url, 
                       chunk.id AS id,
                       score
                ORDER BY score DESC
                LIMIT 20
            """, k=k_vector, query_embedding=query_embedding)
            
            results = []
            seen_ids = set()
            for record in result:
                if record['id'] not in seen_ids:
                    results.append({
                        'text': record['text'],
                        'url': record['url'],
                        'id': record['id'],
                        'score': record.get('score', 0)
                    })
                    seen_ids.add(record['id'])
            
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
            return
        
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️  All Neo4j data deleted")
