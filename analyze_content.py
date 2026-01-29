import sqlite3
import pandas as pd
import os


def analyze_cleaned_content(db_path):
    """Analyze cleaned_content column character counts for all rows."""
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    print(f"📊 Analyzing {db_path}...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables found: {[t[0] for t in tables]}")
    
    # Find the correct table name (might be 'documents' or something else)
    table_name = None
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table[0]});")
        columns = [col[1] for col in cursor.fetchall()]
        if 'cleaned_content' in columns:
            table_name = table[0]
            print(f"✅ Found 'cleaned_content' in table: {table_name}")
            print(f"Columns: {columns}")
            break
    
    if not table_name:
        print("❌ No table with 'cleaned_content' column found")
        conn.close()
        return
    
    # Get total row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    total_rows = cursor.fetchone()[0]
    print(f"\n📈 Total rows in {table_name}: {total_rows:,}")
    
    # Calculate character counts
    print("\n⏳ Calculating character counts...")
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    char_counts = []
    
    for offset in range(0, total_rows, batch_size):
        cursor.execute(f"""
            SELECT 
                LENGTH(COALESCE(cleaned_content, '')) as char_count
            FROM {table_name}
            LIMIT {batch_size} OFFSET {offset}
        """)
        
        batch_counts = [row[0] for row in cursor.fetchall()]
        char_counts.extend(batch_counts)
        
        progress = min(100, ((offset + batch_size) / total_rows) * 100)
  

        print(f" Progress: {progress:.1f}% ({len(char_counts):,}/{total_rows:,})", end='\r')
        import time
        # spinner = ["⠇", "⠙", "⠸", "⠼", "⠦", "⠋", "⠏"]
        # for s in spinner:
        #     print(s, end="\r", flush=True)
        #     time.sleep(0.15)

        proces= ["\\","|","/","-"]
        for i in proces:    
            print(i,end="\r",flush=True)
            time.sleep(0.088)
    print()  # New line after progress
    
    # Calculate statistics
    df = pd.DataFrame({'char_count': char_counts})
    
    print("\n" + "="*60)
    print("📊 CHARACTER COUNT STATISTICS")
    print("="*60)
    print(f"Total rows analyzed: {len(char_counts):,}")
    print(f"Total characters:    {df['char_count'].sum():,}")
    print(f"\nMean:                {df['char_count'].mean():.2f} chars")
    print(f"Median:              {df['char_count'].median():.0f} chars")
    print(f"Min:                 {df['char_count'].min():,} chars")
    print(f"Max:                 {df['char_count'].max():,} chars")
    print(f"Std Dev:             {df['char_count'].std():.2f}")
    
    print(f"\n📈 PERCENTILES")
    print(f"25th percentile:     {df['char_count'].quantile(0.25):.0f} chars")
    print(f"50th percentile:     {df['char_count'].quantile(0.50):.0f} chars")
    print(f"75th percentile:     {df['char_count'].quantile(0.75):.0f} chars")
    print(f"90th percentile:     {df['char_count'].quantile(0.90):.0f} chars")
    print(f"95th percentile:     {df['char_count'].quantile(0.95):.0f} chars")
    print(f"99th percentile:     {df['char_count'].quantile(0.99):.0f} chars")
    
    # Count empty or very small entries
    empty_count = (df['char_count'] == 0).sum()
    small_count = (df['char_count'] < 100).sum()
    
    print(f"\n⚠️  DATA QUALITY")
    print(f"Empty (0 chars):     {empty_count:,} rows ({empty_count/len(char_counts)*100:.2f}%)")
    print(f"Very small (<100):   {small_count:,} rows ({small_count/len(char_counts)*100:.2f}%)")
    
    # Distribution
    print(f"\n📊 DISTRIBUTION")
    bins = [0, 100, 500, 1000, 5000, 10000, 50000, float('inf')]
    labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-50K', '50K+']
    df['range'] = pd.cut(df['char_count'], bins=bins, labels=labels)
    
    for label in labels:
        count = (df['range'] == label).sum()
        pct = count / len(char_counts) * 100
        print(f"{label:12} chars: {count:7,} rows ({pct:5.2f}%)")
    
    # Save detailed results to CSV
    output_csv = f"{db_path}_char_analysis.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Detailed results saved to: {output_csv}")
    
    conn.close()


if __name__ == "__main__":
    # Analyze all databases
    db_files = [
        "databases/VISHALLL.db",
        "databases/SHIVA.db",
        "databases/UDAY.db",
        "databases/MASTER.db",
    ]
    
    # Also check merged database if it exists
    if os.path.exists("databases/merged.db"):
        db_files.insert(0, "databases/merged.db")
    
    for db_file in db_files:
        if os.path.exists(db_file):
            analyze_cleaned_content(db_file)
            print("\n" + "="*60 + "\n")
