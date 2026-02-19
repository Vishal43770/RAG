import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("gemini_api_key"))

for m in client.models.list():
    print(f"Model ID: {m.name}")
