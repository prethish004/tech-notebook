"""
Advanced RAG
HYBRID + RERANKING RAG (Advanced)
Methods:
- Dense retrieval
- Cross-encoder reranking
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

docs = [
    "RAG combines retrieval and generation.",
    "BM25 is lexical search.",
    "Reranking improves relevance."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeds = embedder.encode(docs)

index = faiss.IndexFlatL2(embeds.shape[1])
index.add(embeds)

query = "How does reranking help RAG?"
q_emb = embedder.encode([query])

D, I = index.search(q_emb, 3)
retrieved = [docs[i] for i in I[0]]

# -------- Reranker --------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, doc) for doc in retrieved]
scores = reranker.predict(pairs)

best_doc = retrieved[scores.argmax()]
print("Best Answer Source:", best_doc)
