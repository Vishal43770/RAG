# Test script for Google Gemini API
from google import genai

client = genai.Client(api_key="AIzaSyBWuyUqED8UjnzEV1NSLKWLLn1269W5w4I")

# List available models
models = client.models.list()
print("Available models:")
for model in models:
    print(f"  - {model.name}")
