import os
import pandas as pd
import sqlite3
import ollama
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv
import torch 
from tqdm import tqdm
import numpy as np
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BuildingRag:
    def __init__(self):
        self.neo4j_driver = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print(f"✅ Model loaded on: {device.upper()}")
        print("Using device:", self.embedding_model.device)

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
                source_cursor.execute("SELECT * FROM documents LIMIT ? OFFSET ?", (batch_size, offset))
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
        scene_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=["\n","\n\n",".",",",""],
        )
        dbpath = "databases/merged.db"
        conn = sqlite3.connect(dbpath)
        cursor = conn.cursor()
        if os.path.exists(dbpath):
            pass
        else:
            cursor.execute("""
            CREATE TABLE document_chunks(
                id INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                chunk_text TEXT not NULL,
                embedding BLOB not NULL,
                FOREIGN KEY (id) REFERENCES documents(id)
            ) """)
        cursor.execute(""" select id , cleaned_content from documents;
        """)
        documents=cursor.fetchall()
        for dock_id, text in documents:
            if not text:
                continue
            try:
                chunkid=0
                if len(text)<2000:
                    cursor.execute("""
                    INSERT INTO document_chunks(id, chunk_id, chunk_text)
                    VALUES (?, ?, ?);
                    """, (dock_id, chunkid, text.split("\n")))
                    chunkid+=1
                else:
                    for i in range(0, len(text), 2000):
                        chunk_text=text[i:i+2000]
                        cursor.execute("""
                        INSERT INTO document_chunks(id, chunk_id, chunk_text)
                        VALUES (?, ?, ?);
                        """, (dock_id, chunkid, chunk_text.strip()))
                        chunkid+=1
                    conn.commit()
            except Exception as e:
                print(f"⚠️ Failed to chunk doc {dock_id}: {e}")
                conn.rollback()  # Roll back this document only
        conn.close()
        
    def neo4_setup(self):
        load_dotenv()
        
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "ragpassword")
            )
        )
        with self.neo4j_driver.session() as session:
            session.run("""
             CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)
            session.run("""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
    """)
    def ingest_to_neo4j(self, batch_size=100, limit=None):
        """Load chunks from SQLite to Neo4j with embeddings - RESUMABLE"""
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n📊 Starting Neo4j ingestion using device: {device.upper()}")
        self.neo4_setup()
        conn = sqlite3.connect("databases/merged.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_chunk_rowid INTEGER DEFAULT 0)""")
        cursor.execute("INSERT OR IGNORE INTO ingestion_progress (id, last_chunk_rowid) VALUES (1, 0)")
        conn.commit()
        cursor.execute("SELECT last_chunk_rowid FROM ingestion_progress WHERE id = 1")
        start_rowid = cursor.fetchone()[0]
        if start_rowid > 0:
            print(f"🔄 RESUMING from chunk rowid: {start_rowid:,}")
        else:
            print("🆕 Starting fresh ingestion")    
        query = "SELECT COUNT(*) FROM document_chunks WHERE rowid > ?"
        cursor.execute(query, (start_rowid,))
        total = cursor.fetchone()[0]
        print(f"Total chunks to process: {total:,}")
        if total == 0:
            print("✅ Nothing to process - already complete!")
            conn.close()
            return
        processed = 0
        last_rowid = start_rowid
        try:
            with tqdm(total=total, desc="Ingesting chunks") as pbar:
                while True:
                    # ✅ Fetch embedding column from database
                    query = """
                        SELECT dc.rowid, dc.id, dc.chunk_id, dc.chunk_text, d.url, dc.embedding
                        FROM document_chunks dc
                        LEFT JOIN documents d ON dc.id = d.id
                        WHERE dc.rowid > ?
                        ORDER BY dc.rowid
                        LIMIT ?
                    """
                    cursor.execute(query, (last_rowid, batch_size))
                    batch_rows = cursor.fetchall()
                    if not batch_rows:
                        break
                    
                    # ✅ Separate rows with/without embeddings
                    texts_to_encode = []
                    indices_to_encode = []
                    
                    for i, row in enumerate(batch_rows):
                        chunk_rowid, doc_id, chunk_id, text, url, existing_embedding = row
                        if not text:
                            continue
                        # Check if embedding exists
                        if existing_embedding is None:
                            texts_to_encode.append(text)
                            indices_to_encode.append(i)
                    
                    # ✅ Only generate embeddings if needed
                    new_embeddings = {}
                    if texts_to_encode:
                        print(f"  Generating {len(texts_to_encode)} missing embeddings...")
                        embeddings = self.embedding_model.encode(texts_to_encode, batch_size=48, show_progress_bar=False)
                        new_embeddings = dict(zip(indices_to_encode, embeddings))
                    
                    # ✅ Build Neo4j batch (use existing or new embeddings)
                    neo4j_batch = []
                    for i, row in enumerate(batch_rows):
                        chunk_rowid, doc_id, chunk_id, text, url, existing_embedding = row
                        if not text:
                            continue
                        
                        # Use existing embedding or newly generated one
                        if existing_embedding is not None:
                            # Convert BLOB to list
                            embedding = np.frombuffer(existing_embedding, dtype=np.float32).tolist()
                        else:
                            embedding = new_embeddings[i]
                        if not text:
                            continue
                        neo4j_batch.append({
                            'chunk_rowid': chunk_rowid,
                            'doc_id': doc_id,
                            'chunk_id': chunk_id,
                            'text': text,
                            'url': url or '',
                            'embedding': embedding
                        })
                    if neo4j_batch:
                        self._write_neo4j_batch(neo4j_batch)
                        processed += len(neo4j_batch)
                    last_rowid = batch_rows[-1][0]
                    cursor.execute("UPDATE ingestion_progress SET last_chunk_rowid = ? WHERE id = 1", (last_rowid,))
                    conn.commit()
                    pbar.update(len(batch_rows))
            print(f"\n✅ Ingested {processed:,} chunks to Neo4j")
            print(f"💾 Checkpoint saved at rowid: {last_rowid:,}")
        except Exception as e:
            print(f"\n⚠️ Error occurred: {e}")
            print(f"💾 Progress saved. Can resume from rowid: {last_rowid:,}")
            raise
        finally:
            conn.close()
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



    def build_graph_relationships(self, min_similarity=0.85):
        """Build relationships between chunks based on semantic similarity"""
        print(f"\n🏗️ Building graph relationships (min_similarity={min_similarity})...")
        
        # Get all chunks with embeddings
        query = """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN c.id AS chunk_id, c.embedding AS embedding
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query)
        
        chunks = []
        for record in result:
            chunks.append({
                'id': record['chunk_id'],
                'embedding': record['embedding']
            })
        
        print(f"   Found {len(chunks)} chunks with embeddings")
        
        if len(chunks) < 2:
            print("   Not enough chunks to build relationships")
            return
        
        # Calculate pairwise similarities
        print("   Calculating pairwise similarities...")
        similarities = []
        
        for i in tqdm(range(len(chunks))):
            for j in range(i + 1, len(chunks)):
                chunk1 = chunks[i]
                chunk2 = chunks[j]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(chunk1['embedding'], chunk2['embedding'])
                
                if similarity >= min_similarity:
                    similarities.append({
                        'chunk1_id': chunk1['id'],
                        'chunk2_id': chunk2['id'],
                        'similarity': similarity
                    })
        
        print(f"   Found {len(similarities)} similar pairs")
        
        print("   Creating relationships in Neo4j...")
        with self.neo4j_driver.session() as session:
            for sim in tqdm(similarities, desc="Creating relationships"):
                session.run("""
                    MATCH (c1:Chunk {id: $chunk1_id})
                    MATCH (c2:Chunk {id: $chunk2_id})
                    MERGE (c1)-[r:SIMILAR_TO]->(c2)
                    SET r.similarity = $similarity
                """, chunk1_id=sim['chunk1_id'], chunk2_id=sim['chunk2_id'], similarity=sim['similarity'])
        
        print(f"✅ Built {len(similarities)} relationships")
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return float(dot_product / (norm_vec1 * norm_vec2))
    def create_similarity_edges(self, k=5, threshold=0.7):
        print(f"\n🔗 Creating similarity edges (k={k}, threshold={threshold})...")
        with self.neo4j_driver.session() as session:
                # Get total chunk count
                result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
                total = result.single()['count']
                print(f"Processing {total:,} chunks...")                
                session.run("""
                    MATCH (c1:Chunk)
                    CALL db.index.vector.queryNodes('chunk_embeddings', $k + 1, c1.embedding)
                    YIELD node AS c2, score
                    WHERE c1 < > c2 AND score >= $threshold
                    MERGE (c1)-[r:SIMILAR_TO]-(c2)
                    SET r.score = score """,
                 k=k, threshold=threshold)
                # Count created edges
                result = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS count")
                edge_count = result.single()['count']
                print(f"✅ Created {edge_count:,} similarity edges")

    def fast_search(self, query_text, k=5):
        """Vector-only semantic search in Neo4j using stored embeddings"""
        query_embedding = self.embedding_model.encode(query_text).tolist()

            # Step 2: Run vector similarity query in Neo4j
        with self.neo4j_driver.session() as session:
            result = session.run("""
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_embedding)
            YIELD node, score
            RETURN 
                node.chunk_id AS chunk_id,
                node.text AS text,
                node.url AS url,
                score
            ORDER BY score DESC
        """, k=k, query_embedding=query_embedding)
        return [
            {
                'chunk_id': record['chunk_id'],
                'text': record['text'],
                'url': record['url'],
                'score': float(record['score'])
            }
            for record in result
        ]
    
    def default_search(self, query_text: str = ..., k_vector=5, expand_depth=1, decay_factor=0.85):
        """Hybrid search: vector seeds + graph expansion with decayed scoring"""
        query_embedding = self.embedding_model.encode(query_text).tolist()
        with self.neo4j_driver.session() as session:
            result = session.run(f"""
                // Step 1: Vector search
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
                score = seed_score * (decay_factor ** hop_distance)

                results.append({
                    'text': record['text'],
                    'url': record['url'],
                    'id': chunk_id,
                    'score': round(score, 4)
                })
                seen_ids.add(chunk_id)

            # Optional: sort after applying decay
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
    def close_neo4j(self):
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
            print("🔒 Neo4j connection closed")
    
    def check_neo4j_data_exists(self):
        with self.neo4j_driver.session() as session:
            result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            count = result.single()['count']
            return count > 0
    
    def get_neo4j_stats(self):
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
        print("⚠️  WARNING: This will delete ALL data from Neo4j!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("❌ Cancelled")
            return # Added return to prevent accidental deletion if cancelled
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️  All Neo4j data deleted")
    def ollama_chat(self, query, content):
        prompt = f"""
            You are an assistant.
            Use the context below to answer the question.
            Context:
            {content}
            Question:
            {query}
            Answer clearly and concisely.
            """
    #           response = requests.post(
    #     "http://localhost:11434/api/chat",
    #     json={
    #         "model": "llama3.2:1b",
    #         "messages": [
    #             {"role": "user", "content": prompt}
    #         ],
    #         "stream": False
    #     }
    # )

    # response.raise_for_status()
    #return response.json()["message"]["content"]2
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]

    def main(self):
        self.neo4_setup()
        while True:
            query: str = input("\nEnter your question (type 'exit' to quit): ")

            if query.lower() in ["exit", "bye", "quit"]:
                self.close_neo4j()
                self.clear_neo4j_data()
                print("👋 Goodbye!")
                break  
            mode = input("Enter mode (fast, deep_search, default): ")
            if mode == "fast":
                content = self.fast_search(query)
            elif mode == "deep_search":
                content = self.default_search(
                    query,
                    k_vector=3,
                    expand_depth=2,
                    decay_factor=0.6)
            else:
                content = self.default_search(query)
            response = self.ollama_chat(query, content)
            print("\n🧠 Answer:\n", response)

        
if __name__ == "__main__":
    BuildingRag().main()


