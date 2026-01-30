"""
MULTI-RAG SYSTEM
🧠 MULTI-RAG SYSTEM (Router + Parallel)
Architecture (what this code implements)
User Query
   ↓
Router (Rule / LLM)
   ↓
Selected RAGs (Tech, Legal, General)
   ↓
Parallel Retrieval
   ↓
Reranking (Cross-Encoder)
   ↓
Final LLM Synthesis
📦 Install requirements
pip install faiss-cpu sentence-transformers transformers torch

Features:
- Router-based RAG selection
- Parallel RAG execution
- Cross-encoder reranking
- Final LLM synthesis
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# ---------------------------
# 1️⃣ EMBEDDING & RERANK MODELS
# ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto"
)

# ---------------------------
# 2️⃣ DOCUMENT STORES (Different RAGs)
# ---------------------------
TECH_DOCS = [
    "Python was created by Guido van Rossum.",
    "RAG improves LLM accuracy using retrieval.",
    "FAISS is used for vector similarity search."
]

LEGAL_DOCS = [
    "Copyright laws protect intellectual property.",
    "AI-generated content licensing depends on jurisdiction.",
    "Data privacy laws regulate personal data usage."
]

GENERAL_DOCS = [
    "Artificial Intelligence mimics human intelligence.",
    "Large Language Models generate text probabilistically.",
    "Retrieval-Augmented Generation combines search and generation."
]

# ---------------------------
# 3️⃣ BUILD VECTOR INDEX
# ---------------------------
def build_index(docs):
    embeds = embedder.encode(docs)
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    return index, embeds

tech_index, tech_embeds = build_index(TECH_DOCS)
legal_index, legal_embeds = build_index(LEGAL_DOCS)
general_index, general_embeds = build_index(GENERAL_DOCS)

# ---------------------------
# 4️⃣ RETRIEVER FUNCTION
# ---------------------------
def retrieve(query, docs, index, k=2):
    q_emb = embedder.encode([query])
    _, I = index.search(q_emb, k)
    return [docs[i] for i in I[0]]

# ---------------------------
# 5️⃣ RAG FUNCTIONS
# ---------------------------
def tech_rag(query):
    return retrieve(query, TECH_DOCS, tech_index)

def legal_rag(query):
    return retrieve(query, LEGAL_DOCS, legal_index)

def general_rag(query):
    return retrieve(query, GENERAL_DOCS, general_index)

# ---------------------------
# 6️⃣ ROUTER (Rule based – can upgrade to LLM)
# ---------------------------
def router(query):
    query_lower = query.lower()
    selected = []

    if any(word in query_lower for word in ["law", "license", "copyright"]):
        selected.append(legal_rag)

    if any(word in query_lower for word in ["python", "ai", "rag", "model"]):
        selected.append(tech_rag)

    if not selected:
        selected.append(general_rag)

    return selected

# ---------------------------
# 7️⃣ PARALLEL RAG EXECUTION
# ---------------------------
def parallel_rag(query, rag_functions):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(rag, query) for rag in rag_functions]
        for future in futures:
            results.extend(future.result())
    return results

# ---------------------------
# 8️⃣ RERANK RESULTS
# ---------------------------
def rerank_docs(query, docs):
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), reverse=True)
    return [doc for _, doc in ranked[:3]]

# ---------------------------
# 9️⃣ FINAL ANSWER GENERATION
# ---------------------------
def generate_answer(query, context_docs):
    context = "\n".join(context_docs)

    prompt = f"""
You are an expert assistant.
Answer ONLY using the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    out = llm(prompt, max_new_tokens=200)
    return out[0]["generated_text"]

# ---------------------------
# 🔟 MAIN PIPELINE
# ---------------------------
def multi_rag_pipeline(query):
    rag_functions = router(query)
    retrieved_docs = parallel_rag(query, rag_functions)
    reranked_docs = rerank_docs(query, retrieved_docs)
    answer = generate_answer(query, reranked_docs)
    return answer

# ---------------------------
# 🧪 TEST
# ---------------------------
if __name__ == "__main__":
    query = "Is AI-generated content protected by copyright law?"
    print("\nUSER QUESTION:\n", query)
    print("\nFINAL ANSWER:\n")
    print(multi_rag_pipeline(query))
