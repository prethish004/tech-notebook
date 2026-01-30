"""
Chunked 
CHUNKING + METADATA RAG (Production Style)
Methods:
- Chunking
- Metadata handling
- Prompt control
"""

import faiss
from sentence_transformers import SentenceTransformer

text = """
RAG allows LLMs to use external documents.
It reduces hallucination.
It is widely used in enterprise AI.
"""

def chunk_text(txt, size=15):
    words = txt.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

chunks = chunk_text(text)
metadata = [{"source": "doc1", "chunk_id": i} for i in range(len(chunks))]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeds = embedder.encode(chunks)

index = faiss.IndexFlatL2(embeds.shape[1])
index.add(embeds)

query = "Why is RAG useful?"
q_emb = embedder.encode([query])

D, I = index.search(q_emb, 2)

for i in I[0]:
    print("TEXT:", chunks[i])
    print("META:", metadata[i])
