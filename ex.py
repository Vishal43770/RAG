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












































