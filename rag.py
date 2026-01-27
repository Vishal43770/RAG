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
        
    def embedding(self):
        pass

if __name__ == "__main__":
    BuildingRag().know_data()
