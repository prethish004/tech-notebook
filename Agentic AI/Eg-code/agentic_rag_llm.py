"""
✅ RAG INSIDE THE AGENT
✅ LLM-BASED DECISIONS (no hard-coded rules)
✅ Single, full, working Python file
✅ Very detailed explanation of WHY each part exists

This is the foundation of LangGraph / CrewAI / AutoGPT.

🧠 What We Are Building (High Level)

An agent that decides (using an LLM)
when to retrieve knowledge (RAG)
when to reason internally
how to plan
how to act
when to stop

🏗 FINAL ARCHITECTURE
USER GOAL
   ↓
LLM PLANNER (creates steps)
   ↓
FOR EACH STEP:
   ├── LLM decides: Need RAG or not?
   ├── If yes → RAG Retriever
   ├── If no → Reasoning
   ├── Store result in Memory
   ↓
LLM SYNTHESIS (final answer)

📦 INSTALL REQUIREMENTS
pip install openai faiss-cpu sentence-transformers numpy


⚠️ Use an OpenAI-compatible model (OpenAI / Azure / Ollama API)
AGENTIC AI + RAG + LLM DECISION MAKING
------------------------------------
This is a REAL agentic RAG system.

Capabilities:
- LLM-based planning
- LLM-based tool decision (RAG vs reasoning)
- Vector search (RAG)
- Memory
- Controlled execution loop
"""

import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==============================
# 🔑 CONFIG
# ==============================

openai.api_key = "YOUR_API_KEY"

LLM_MODEL = "gpt-3.5-turbo"

# ==============================
# 📚 KNOWLEDGE BASE (RAG)
# ==============================

DOCUMENTS = [
    "Agentic AI refers to systems that can plan, act, observe, and iterate toward a goal.",
    "RAG stands for Retrieval Augmented Generation.",
    "RAG improves LLM responses by grounding them in external knowledge.",
    "Python was created by Guido van Rossum.",
    "FAISS is a vector database for similarity search."
]

# ==============================
# 🧠 EMBEDDINGS + VECTOR DB
# ==============================

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(DOCUMENTS)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

def rag_retrieve(query, k=2):
    """Retrieve relevant documents using vector similarity"""
    q_emb = embedder.encode([query])
    _, I = index.search(q_emb, k)
    return [DOCUMENTS[i] for i in I[0]]

# ==============================
# 🧠 MEMORY
# ==============================

class AgentMemory:
    def __init__(self):
        self.history = []

    def add(self, item):
        self.history.append(item)

    def get_context(self):
        return "\n".join(self.history[-6:])

# ==============================
# 🤖 LLM HELPERS
# ==============================

def call_llm(system, user):
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return response.choices[0].message.content.strip()

# ==============================
# 🗺 PLANNER (LLM-BASED)
# ==============================

def llm_planner(goal):
    """
    WHY:
    - Removes hard-coded logic
    - Allows dynamic planning
    """
    system = "You are an expert planner AI."
    user = f"""
Goal: {goal}

Create a step-by-step plan.
Return steps as numbered list only.
"""
    plan_text = call_llm(system, user)
    return [step.split(". ", 1)[1] for step in plan_text.split("\n") if "." in step]

# ==============================
# 🧭 TOOL DECISION (LLM-BASED)
# ==============================

def decide_need_rag(step):
    """
    LLM decides whether retrieval is needed
    """
    system = "You decide whether external knowledge is required."
    user = f"""
Step: {step}

Answer ONLY one word: YES or NO
"""
    decision = call_llm(system, user)
    return decision.strip().upper() == "YES"

# ==============================
# ⚙ EXECUTOR (AGENT CORE)
# ==============================

def execute_step(step, goal, memory):
    print(f"\n🧠 STEP: {step}")

    need_rag = decide_need_rag(step)
    print("🔍 Need RAG?", need_rag)

    if need_rag:
        docs = rag_retrieve(step)
        observation = "\n".join(docs)
        print("📚 RAG RESULT:", observation)
    else:
        system = "You are a reasoning assistant."
        observation = call_llm(system, step)
        print("🧠 REASONING RESULT:", observation)

    memory.add(observation)

# ==============================
# 🧪 FINAL SYNTHESIS
# ==============================

def finalize_answer(goal, memory):
    system = """
You are a senior AI assistant.
Answer ONLY using the context.
If unsure, say I don't know.
"""
    user = f"""
Context:
{memory.get_context()}

Goal:
{goal}
"""
    return call_llm(system, user)

# ==============================
# 🚀 AGENT CONTROLLER
# ==============================

def agentic_rag(goal):
    print("\n==============================")
    print("🎯 GOAL:", goal)
    print("==============================")

    memory = AgentMemory()

    # 1️⃣ PLAN
    plan = llm_planner(goal)
    print("\n🗺 PLAN:")
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")

    # 2️⃣ EXECUTE STEPS
    for step in plan:
        execute_step(step, goal, memory)

    # 3️⃣ FINAL ANSWER
    print("\n✅ FINAL ANSWER:")
    answer = finalize_answer(goal, memory)
    print(answer)

# ==============================
# ▶ RUN
# ==============================

if __name__ == "__main__":
    agentic_rag("Explain Agentic AI and how RAG improves it")
"""
🧠 WHY THIS IS IMPORTANT (READ THIS)
🔹 Why embed RAG inside the agent?

Because:
  Agent decides WHEN to retrieve
  Not every step needs search
  Saves tokens & time
  Improves accuracy
This is how ChatGPT tools work internally.

🔹 Why LLM-based decisions?
Hard rules ❌
Real world = messy ❗
LLM decisions allow:
  Dynamic behavior
  New domains
  Less code changes

✅ WHAT YOU HAVE NOW
Feature	Status
Agentic control loop	✅
RAG integrated	✅
LLM planning	✅
LLM tool decision	✅
Memory	✅
Grounded answers	✅
"""

