"""
FAISS RAG
FAISS VECTOR DB RAG (Standard Industry Setup)
Methods:
- Vector DB (FAISS)
- Top-K retrieval
- Open-source LLM
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

docs = [
    "RAG enables private data usage with LLMs.",
    "Vector databases store embeddings.",
    "FAISS is used for fast similarity search."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeds = embedder.encode(docs)

# -------- FAISS --------
dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeds)

query = "How does FAISS help RAG?"
q_emb = embedder.encode([query])

D, I = index.search(q_emb, k=2)
context = "\n".join([docs[i] for i in I[0]])

# -------- LLM --------
llm = pipeline("text-generation",
               model="mistralai/Mistral-7B-Instruct-v0.2",
               device_map="auto")

prompt = f"""
Answer using only context.

Context:
{context}

Question:
{query}
"""

out = llm(prompt, max_new_tokens=150)
print(out[0]["generated_text"])
