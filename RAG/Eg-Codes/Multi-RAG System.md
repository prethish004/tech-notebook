# 🧠 Multi-RAG System (Router + Parallel Retrieval)

A **production-style Retrieval-Augmented Generation (RAG)** architecture that connects **multiple RAG pipelines** using a **router**, executes them **in parallel**, **reranks results**, and generates a **grounded final answer** using an LLM.

This project demonstrates **how real enterprise-grade RAG systems are built**.

---

## 🚀 What This Project Does

Instead of using a single RAG system, this project:

✅ Routes user queries to **multiple domain-specific RAGs**  
✅ Runs selected RAGs **in parallel**  
✅ **Reranks retrieved documents** for best relevance  
✅ Sends **clean context** to an LLM  
✅ Prevents hallucinations using **strict prompting**

---

## 🧩 Architecture Overview

```

User Question
↓
Router (Rule-based / Extendable to LLM)
↓
Selected RAG Pipelines
(Tech | Legal | General)
↓
Parallel Retrieval (Threading)
↓
Cross-Encoder Reranking
↓
Final LLM Answer (Context-grounded)

```

---

## 📁 Project Structure

```

multi_rag/
│
├── multi_rag.py     # Complete Multi-RAG pipeline
└── README.md        # Documentation

````

---

## 🧠 RAG Pipelines Included

| RAG Name | Purpose |
|--------|--------|
Tech RAG | Technical knowledge (AI, Python, RAG, FAISS)
Legal RAG | Legal concepts (copyright, data privacy, licensing)
General RAG | Common AI and LLM concepts

Each RAG has:
- Its **own documents**
- Its **own FAISS index**
- Independent retrieval logic

---

## 🛠 Technologies Used

- **Python 3.8+**
- **FAISS** – Vector similarity search
- **Sentence Transformers** – Embeddings & reranking
- **Cross-Encoder** – Relevance reranking
- **Hugging Face Transformers** – LLM inference
- **ThreadPoolExecutor** – Parallel execution

---

## 📦 Installation

```bash
pip install faiss-cpu sentence-transformers transformers torch
````

> ⚠️ For GPU usage, ensure CUDA-compatible PyTorch is installed.

---

## 🔧 How It Works (Step-by-Step)

### 1️⃣ Embedding Creation

Each RAG converts its documents into dense vector embeddings using:

```
all-MiniLM-L6-v2
```

---

### 2️⃣ Vector Indexing (FAISS)

Each document set is stored in a **separate FAISS index**, allowing:

* Fast similarity search
* Domain isolation

---

### 3️⃣ Router (Decision Layer)

The router inspects the query and selects the relevant RAGs.

Example:

* `"copyright"` → Legal RAG
* `"AI", "Python"` → Tech RAG
* Unknown → General RAG

> 🔁 Router can be upgraded to **LLM-based routing**

---

### 4️⃣ Parallel Retrieval

Selected RAGs execute **simultaneously**, reducing latency.

```python
ThreadPoolExecutor
```

---

### 5️⃣ Reranking (Critical Step)

All retrieved documents are reranked using a **cross-encoder** to ensure maximum relevance.

Model used:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This step significantly improves RAG accuracy.

---

### 6️⃣ Final Answer Generation

The LLM receives:

* User query
* Top-ranked contextual documents

Strict prompt rules:

* ❌ No hallucination
* ❌ No outside knowledge
* ✅ Context-only answers

---

## ▶️ How to Run

```bash
python multi_rag.py
```

### Example Query

```text
Is AI-generated content protected by copyright law?
```

### Output

```text
A grounded answer synthesized from legal and technical sources.
```

---

## 🧪 Example Use Cases

* Enterprise internal search
* Legal + technical AI assistants
* Medical + policy question answering
* Research assistants
* Multi-domain chatbots

---

## 🚫 When NOT to Use RAG

❌ Pure reasoning or math
❌ Creative writing
❌ Code generation
❌ Opinion-based answers

✔ Use RAG for **knowledge-grounded answers**

---

## 🚀 Extension Ideas

You can extend this system with:

* 🔄 LLM-based Router
* 📄 PDF / CSV / SQL RAG
* 🌐 Web-search RAG
* 🧠 LangGraph or CrewAI
* 📊 RAG evaluation (RAGAS)
* 🖥 Streamlit UI
* 🔒 Authentication & logging

---

## 🧠 Key Design Principle

> **RAGs do not talk directly.**
> They communicate via **clean, retrieved, reranked context**.

---

## 📜 License

MIT License
Free to use, modify, and distribute.

---

## ⭐ Final Note

This is **not a toy RAG**.
This is the **same conceptual architecture used in real-world enterprise AI systems**.

If you understand this project, you understand **RAG at production level** ✅

```

