#!/bin/bash

# Neo4j Start Script for RAG System

echo "🔍 Checking Neo4j status..."

if docker ps | grep -q neo4j-rag; then
    echo "✅ Neo4j is already running!"
else
    echo "⚠️  Neo4j is not running. Starting it now..."
    docker start neo4j-rag
    
    echo "⏳ Waiting for Neo4j to be ready..."
    sleep 8
    
    if docker ps | grep -q neo4j-rag; then
        echo "✅ Neo4j started successfully!"
    else
        echo "❌ Failed to start Neo4j. Please check Docker."
        exit 1
    fi
fi

echo ""
echo "🚀 Starting RAG chat system..."
echo ""

# Run the Python script
python3 ex.py
