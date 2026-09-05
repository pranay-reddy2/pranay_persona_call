"""Retrieval over Pinecone with Gemini embeddings.

Both the Pinecone client and the index handle are created lazily on the
first query. Creating them at import time cost ~3 s of boot (the Pinecone
SDK describes the index over the network on construction), which is most of
what made this service slow to come up on Render.
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "scaler")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

_index = None


def _get_index():
    """Pinecone client + index, created once on first use."""
    global _index
    if _index is None:
        from pinecone import Pinecone  # heavy import, deliberately lazy
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(PINECONE_INDEX)
    return _index


def embed(text, task_type="RETRIEVAL_QUERY"):
    key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (free key: https://aistudio.google.com/apikey)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"
    body = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": EMBED_DIM,
    }
    last = None
    for attempt in range(3):
        res = requests.post(url, json=body, headers={"x-goog-api-key": key}, timeout=30)
        if res.status_code in (429, 500, 503):
            last = res
            time.sleep(1.5 * (attempt + 1))
            continue
        res.raise_for_status()
        return res.json()["embedding"]["values"]
    last.raise_for_status()


def retrieve(query, top_k=4):
    query_vector = embed(query)
    results = _get_index().query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    chunks = [match["metadata"]["text"] for match in results["matches"]]
    return "\n\n---\n\n".join(chunks)
