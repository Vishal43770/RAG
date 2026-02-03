"""
Database Viewer for document_chunks
====================================
Shows table schema (columns) and first 5 rows
"""

import sqlite3
import pandas as pd


def view_document_chunks(db_path="databases/merged.db"):
    """
    Display document_chunks table information:
    - Column names and types
    - First 5 rows of data
    """
    
    print("="*70)
    print("  DOCUMENT CHUNKS TABLE VIEWER")
    print("="*70)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # ========================================
    # 1. Show Table Schema (Columns)
    # ========================================
    print("\n📋 TABLE SCHEMA (Columns)")
    print("-" * 70)
    
    cursor.execute("PRAGMA table_info(document_chunks);")
    columns = cursor.fetchall()
    
    if not columns:
        print("⚠️ Table 'document_chunks' does not exist!")
        conn.close()
        return
    
    print(f"\n{'#':<5} {'Column Name':<20} {'Type':<15} {'Not Null':<10} {'Default'}")
    print("-" * 70)
    
    for col in columns:
        col_id, name, dtype, not_null, default_val, pk = col
        not_null_str = "YES" if not_null else "NO"
        default_str = str(default_val) if default_val else "-"
        print(f"{col_id:<5} {name:<20} {dtype:<15} {not_null_str:<10} {default_str}")
    
    # ========================================
    # 2. Show Row Count
    # ========================================
    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    total_rows = cursor.fetchone()[0]
    print(f"\n📊 Total Rows: {total_rows:,}")
    
    # ========================================
    # 3. Show First 5 Rows
    # ========================================
    print("\n📄 FIRST 5 ROWS")
    print("-" * 70)
    
    # Get column names for display
    column_names = [col[1] for col in columns]
    
    # Fetch first 5 rows
    query = "SELECT * FROM document_chunks LIMIT 5"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("⚠️ No data in table")
    else:
        # Display with pandas for nice formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)  # Truncate long text
        
        print(df.to_string(index=False))
    
    # ========================================
    # 4. Show Sample Statistics
    # ========================================


    print("\n\n📈 STATISTICS")
    print("-" * 70)
    
    # Check if chunk_text column exists
    if 'chunk_text' in column_names:
        cursor.execute("""
            SELECT 
                MIN(LENGTH(chunk_text)) as min_length,
                MAX(LENGTH(chunk_text)) as max_length,
                AVG(LENGTH(chunk_text)) as avg_length
            FROM document_chunks
            WHERE chunk_text IS NOT NULL
        """)
        stats = cursor.fetchone()
        if stats:
            print(f"Chunk Text Length:")
            print(f"  Min: {stats[0]:,} chars")
            print(f"  Max: {stats[1]:,} chars")
            print(f"  Avg: {stats[2]:,.1f} chars")
    
    # Check unique documents
    if 'id' in column_names:
        cursor.execute("SELECT COUNT(DISTINCT id) FROM document_chunks")
        unique_docs = cursor.fetchone()[0]
        print(f"\nUnique Documents: {unique_docs:,}")
    
    conn.close()
    print("\n" + "="*70)






def view_all_tables(db_path="databases/merged.db"):
    """Show all tables in the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n📁 ALL TABLES IN DATABASE")
    print("-" * 70)
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    for i, (table_name,) in enumerate(tables, 1):
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"{i}. {table_name:<30} ({row_count:,} rows)")
    
    conn.close()


if __name__ == "__main__":
    import sys
    
    # Get database path from arguments or use default
    db_path = sys.argv[1] if len(sys.argv) > 1 else "databases/merged.db"
    
    try:
        # Show document_chunks details
        view_document_chunks(db_path)
        
        # Show all tables summary
        view_all_tables(db_path)
        
    except FileNotFoundError:
        print(f"❌ Database not found: {db_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
