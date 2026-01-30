"""
Agentic RAG
AGENTIC RAG (Tool-Using RAG)
Methods:
- LLM decides when to retrieve
- Tool usage
- Multi-step reasoning
"""

from sentence_transformers import SentenceTransformer
import faiss

knowledge = [
    "Python was created by Guido van Rossum.",
    "RAG helps LLMs access external knowledge."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeds = embedder.encode(knowledge)

index = faiss.IndexFlatL2(embeds.shape[1])
index.add(embeds)

def retriever(question):
    q_emb = embedder.encode([question])
    _, I = index.search(q_emb, 1)
    return knowledge[I[0][0]]

def agent(question):
    if "who" in question.lower() or "what" in question.lower():
        context = retriever(question)
        return f"Answer (from tool): {context}"
    else:
        return "Answer (LLM only): Reasoning response"

print(agent("Who created Python?"))
print(agent("Explain benefits of RAG"))
