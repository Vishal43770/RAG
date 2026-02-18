# Test script for Google Gemini API
from google import genai

client = genai.Client(api_key="AIzaSyBWuyUqED8UjnzEV1NSLKWLLn1269W5w4I")
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Hello, how are you?'
)
print(response.text)
    