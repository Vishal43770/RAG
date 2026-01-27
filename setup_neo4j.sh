#!/bin/bash
# Neo4j Docker Setup for Graph RAG

echo "🚀 Starting Neo4j in Docker..."

docker run \
    --name neo4j-rag \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/ragpassword \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v neo4j_data:/data \
    neo4j:latest

echo ""
echo "✅ Neo4j started successfully!"
echo ""
echo "📊 Access Neo4j Browser: http://localhost:7474"
echo "🔌 Bolt Protocol: bolt://localhost:7687"
echo "👤 Username: neo4j"
echo "🔑 Password: ragpassword"
echo ""
echo "⏳ Wait ~30 seconds for Neo4j to fully start, then visit the browser URL"
