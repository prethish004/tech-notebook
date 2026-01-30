"""
BASIC RAG
BASIC RAG (No Vector DB – Scratch Logic)
Methods used:
- Embeddings
- Cosine similarity
- Context injection
- LLM answering
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import openai

# -------- Data --------
docs = [
    "RAG improves LLMs using external knowledge.",
    "RAG stands for Retrieval Augmented Generation.",
    "RAG is used in enterprise question answering."
]

# -------- Embedding --------
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)

# -------- Similarity --------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "What is RAG?"
query_emb = model.encode([query])[0]

scores = [cosine_sim(query_emb, d) for d in doc_embeddings]
best_doc = docs[np.argmax(scores)]

# -------- LLM Call --------
prompt = f"""
Context:
{best_doc}

Question:
{query}
"""

openai.api_key = "YOUR_API_KEY"

res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user","content": prompt}]
)

print(res.choices[0].message.content)
