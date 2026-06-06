import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("scaler")


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def embed(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
    res = requests.post(url, json={
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": 768
    })
    res.raise_for_status()
    return res.json()["embedding"]["values"]

def upsert_chunks(chunks, source):
    vectors = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 20:  # skip tiny/empty chunks
            continue
        
        # Prepend source context
        if source.startswith("github-"):
            repo_name = source.replace("github-", "")
            enriched_chunk = f"[From GitHub repo: {repo_name}]\n{chunk}"
        elif source == "resume":
            enriched_chunk = f"[From Pranay's resume]\n{chunk}"
        elif source == "persona":
            enriched_chunk = f"[From Pranay's persona facts]\n{chunk}"
        else:
            enriched_chunk = chunk

        vector = embed(enriched_chunk)
        vectors.append({
            "id": f"{source}-{i}",
            "values": vector,
            "metadata": {"text": enriched_chunk, "source": source}
        })

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    
    print(f"✅ Upserted {len(vectors)} chunks from {source}")


# Resume
# ── Persona doc ──────────────────────────────────────────
with open("data/persona.txt", "r", encoding="utf-8") as f:
    persona_text = f.read()

persona_chunks = chunk_text(persona_text, chunk_size=200, overlap=30)
upsert_chunks(persona_chunks, "persona")

# ── Resume ───────────────────────────────────────────────
with open("data/resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Split by section instead of fixed word count
sections = resume_text.split("\n\n")
resume_chunks = []
for section in sections:
    section = section.strip()
    if len(section) > 30:  # skip empty/tiny sections
        resume_chunks.append(section)

upsert_chunks(resume_chunks, "resume")


# GitHub READMEs
GITHUB_USERNAME = "pranay-reddy2"

REPOS = [
    "NoteBookLLM",
    "NoteBookLLM_Backend", 
    "CLI-Agent",
    "collabify",
    "GYM_CRM",
    "Trading_Bot",
    "LensPDF",
    "vocallabs",
    "google-calender"
]

for repo in REPOS:
    found = False

    for branch in ["main", "master"]:
        url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{repo}/{branch}/README.md"

        try:
            res = requests.get(url, timeout=10)

            if res.status_code == 200:
                chunks = chunk_text(res.text)
                upsert_chunks(chunks, f"github-{repo}")
                found = True
                break

        except Exception as e:
            print(f"Error fetching {repo}: {e}")

    if not found:
        print(f"⚠️ Could not fetch README for {repo}")

print("\n✅ Ingestion complete")