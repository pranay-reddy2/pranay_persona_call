import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("scaler")

def embed(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
    res = requests.post(url, json={
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768
    })
    res.raise_for_status()
    return res.json()["embedding"]["values"]

def retrieve(query, top_k=4):
    query_vector = embed(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    chunks = [match["metadata"]["text"] for match in results["matches"]]
    return "\n\n---\n\n".join(chunks)