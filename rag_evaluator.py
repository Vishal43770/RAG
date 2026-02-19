import os
import time
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from ex import BuildingRag

# Load environment variables
load_dotenv()

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Evaluator")

class RAGEvaluator:
    def __init__(self):
        self.rag = BuildingRag()
        self.rag.setup_neo4j_graph()
        
        # Initialize Groq client
        api_key = os.getenv("groq")
        if not api_key:
            raise ValueError("❌ groq api key not found in .env")
            
        self.client = Groq(api_key=api_key)
        self.model_id = "llama-3.3-70b-versatile"
        logger.info(f"✅ Evaluator initialized with Groq ({self.model_id})")

    def evaluate_response(self, query, context, response):
        """
        Uses Groq to score the RAG Triad.
        Returns scores for Faithfulness, Relevance, and Context Precision.
        """
        prompt = f"""
        You are an expert RAG (Retrieval-Augmented Generation) evaluator. 
        Your task is to judge the quality of a RAG system's response based on the provided context.

        ### INPUTS:
        - **USER QUESTION**: {query}
        - **RETRIVED CONTEXT**: {context}
        - **SYSTEM RESPONSE**: {response}

        ### EVALUATION METRICS (Score 0.0 to 1.0):
        1. **Faithfulness**: Is the SYSTEM RESPONSE derived ONLY from the RETRIEVED CONTEXT? (1.0 = perfect, 0.0 = contains hallucinations or external info).
        2. **Answer Relevance**: Does the SYSTEM RESPONSE actually answer the USER QUESTION? (1.0 = direct answer, 0.0 = irrelevant).
        3. **Context Precision**: How much of the RETRIEVED CONTEXT was actually useful for answering the question? (1.0 = all context used, 0.0 = context was useless).

        ### OUTPUT FORMAT:
        Return ONLY a JSON object with the following keys:
        {{
            "faithfulness": float,
            "relevance": float,
            "precision": float,
            "reasoning": "Brief explanation for the scores"
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"❌ Groq Eval Error: {e}")
            return None

    def run_benchmark(self, test_dataset):
        results = []
        print("\n" + "="*70)
        print("🧪 STARTING QUANTITATIVE RAG BENCHMARK (GROQ JUDGE)")
        print("="*70)

        for i, query in enumerate(test_dataset, 1):
            print(f"\n[{i}/{len(test_dataset)}] Evaluating: '{query[:50]}...'")
            
            # 1. Retrieve & Generate using existing pipeline
            start_time = time.time()
            content_chunks = self.rag.default_search(query, k_vector=25, expand_depth=2, decay_factor=0.6)
            
            if not content_chunks:
                print("   ⚠️ No chunks found. Skipping metrics.")
                continue

            # Limit to top 10 as per current deep search logic
            top_chunks = content_chunks[:10]
            context_text = "\n\n".join([f"Source: {c.get('url', 'N/A')}\nContent: {c['text']}" for c in top_chunks])
            
            ai_response = self.rag.ollama_chat(query, top_chunks)
            total_time = time.time() - start_time

            # 2. Get Metrics from Groq
            scores = self.evaluate_response(query, context_text, ai_response)
            
            if scores:
                results.append({
                    "query": query,
                    "faithfulness": scores.get('faithfulness', 0.0),
                    "relevance": scores.get('relevance', 0.0),
                    "precision": scores.get('precision', 0.0),
                    "latency": round(total_time, 2),
                    "reasoning": scores.get('reasoning', "N/A")
                })
                print(f"   ✅ Scores: F:{scores.get('faithfulness')} | R:{scores.get('relevance')} | P:{scores.get('precision')}")
            else:
                print("   ❌ Evaluation failed.")

        # 3. Summary Report
        if not results:
            print("❌ No results to show.")
            return

        df = pd.DataFrame(results)
        print("\n" + "="*70)
        print("📊 FINAL EVALUATION REPORT")
        print("="*70)
        print(f"Total Queries Tested : {len(results)}")
        print(f"Avg Faithfulness      : {df['faithfulness'].mean():.2f}")
        print(f"Avg Answer Relevance  : {df['relevance'].mean():.2f}")
        print(f"Avg Context Precision : {df['precision'].mean():.2f}")
        print(f"Avg Latency           : {df['latency'].mean():.2f}s")
        print("="*70)
        
        # Save to CSV
        output_file = "rag_metrics_report.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Detailed report saved to: {output_file}\n")

if __name__ == "__main__":
    test_questions = [
        "What is the procedure for copying migration projects?",
        "How can I manage returnable packaging logistics in S/4HANA?",
        "What are the search categories for packing instructions?",
        "Can I create text items in packing instructions?",
        "How do I track movements to returnable packaging accounts?"
    ]
    
    try:
        evaluator = RAGEvaluator()
        evaluator.run_benchmark(test_questions)
    except Exception as e:
        print(f"❌ Failed to run evaluator: {e}")
    finally:
        if 'evaluator' in locals() and hasattr(evaluator, 'rag'):
            evaluator.rag.close_neo4j()
