# RAG System Quick Start Guide

## ✅ Neo4j is Now Running!

Your Neo4j database is up and running on:
- **Bolt**: `localhost:7687`
- **Browser**: `http://localhost:7474`

---

## How to Use

### Option 1: Auto-start Script (Recommended)
```bash
./start_rag.sh
```
This automatically checks and starts Neo4j if needed, then launches the chat.

### Option 2: Manual Start
```bash
python3 ex.py
```

Choose from the menu:
- **Option 1**: Run full pipeline (first time setup)
- **Option 2**: Connect to existing Neo4j data
- **Option 3**: Just chat (if already set up)

---

## Common Commands

### Start Neo4j
```bash
docker start neo4j-rag
```

### Stop Neo4j
```bash
docker stop neo4j-rag
```

### Check Neo4j Status
```bash
docker ps | grep neo4j
```

### Neo4j Logs
```bash
docker logs neo4j-rag
```

---

## Troubleshooting

### "Connection refused" error
Neo4j is not running. Start it with:
```bash
docker start neo4j-rag
```

### Neo4j not starting
Check logs:
```bash
docker logs neo4j-rag --tail 50
```

---

## Ready to Chat!

Neo4j is running. You can now:
```bash
python3 ex.py
```

Then select option **3** for quick chat or **2** to verify data exists.
