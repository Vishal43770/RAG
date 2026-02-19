import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

passwords = ["password", "ragpassword"]
for p in passwords:
    print(f"Testing password: {p}")
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", p))
        with driver.session() as session:
            session.run("RETURN 1")
        print(f"✅ Success with password: {p}")
        driver.close()
        break
    except Exception as e:
        print(f"❌ Failed with password {p}: {e}")
