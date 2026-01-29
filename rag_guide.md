# RAG (Retrieval-Augmented Generation) Complete Guide

## Table of Contents
1. [What is RAG?](#what-is-rag)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Key Components](#key-components)
5. [RAG Workflow](#rag-workflow)
6. [Popular Libraries & Tools](#popular-libraries--tools)
7. [Implementation Examples](#implementation-examples)
8. [LangChain RAG Example](#langchain-rag-example)
9. [LlamaIndex RAG Example](#llamaindex-rag-example)
10. [Vector Databases](#vector-databases)
11. [Embeddings](#embeddings)
12. [Best Practices](#best-practices)
13. [Performance Optimization](#performance-optimization)
14. [Common Use Cases](#common-use-cases)

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It's a technique that combines:
1. **Retrieval:** Finding relevant information from external data sources
2. **Augmentation:** Adding this information to the LLM's prompt
3. **Generation:** Using the LLM to generate responses based on both its training and the retrieved context

### Why Use RAG?

- **Reduces Hallucinations:** LLMs have access to factual data
- **Up-to-date Information:** Can retrieve real-time or recent data
- **Domain-Specific Knowledge:** Works with private or specialized documents
- **Cost-Effective:** Don't need to fine-tune large models
- **Better Accuracy:** Provides sources and citations
- **Scalability:** Works with large document bases

### RAG vs Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Speed** | Fast to implement | Requires training time |
| **Cost** | Lower | Higher (GPU intensive) |
| **Updates** | Easy (add new docs) | Need to retrain |
| **Knowledge** | Specific to documents | Baked into model |
| **Use Case** | Document Q&A, Search | Task-specific behavior |

---

## Core Concepts

### 1. Embeddings
Embeddings convert text into numerical vectors that capture semantic meaning.

**Example:**
```
"The cat sat on the mat" → [0.23, -0.45, 0.67, 0.12, ...]
"A feline was on a rug"  → [0.22, -0.44, 0.68, 0.13, ...]
```

These vectors are similar because the sentences have similar meanings.

### 2. Vector Database
Stores embeddings and allows similarity search to find relevant documents.

**Popular options:**
- Pinecone
- Weaviate
- FAISS
- Milvus
- ChromaDB
- Qdrant

### 3. Similarity Search
Finds documents most similar to the user's question using vector distance metrics.

**Distance metrics:**
- **Cosine Similarity:** Most common, measures angle between vectors
- **Euclidean Distance:** Straight-line distance
- **Manhattan Distance:** Grid-based distance

### 4. Prompt Engineering
Crafting prompts that tell the LLM how to use retrieved context.

### 5. Reranking
Re-ordering retrieved documents by relevance before feeding to LLM.

---

## Architecture Overview

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Convert to Embedding    │
│ (Embedding Model)       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Vector Database        │
│  Similarity Search      │
│  (Find Top-K docs)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Retrieved Documents    │
│  (Context)              │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Build Prompt                    │
│  Query + Context + Instructions  │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  LLM (GPT, Claude)   │
│  Generate Response   │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Final Answer        │
│  + Sources           │
└──────────────────────┘
```

---

## Key Components

### 1. Document Loaders
Load documents from various sources.

```python
from langchain.document_loaders import PDFLoader, TextLoader
from langchain.document_loaders import DirectoryLoader

# Load single PDF
loader = PDFLoader("document.pdf")
documents = loader.load()

# Load all text files from directory
loader = DirectoryLoader("./documents", glob="**/*.txt")
documents = loader.load()
```

### 2. Text Splitters
Break documents into manageable chunks.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Size of each chunk
    chunk_overlap=200       # Overlap between chunks
)

chunks = splitter.split_documents(documents)
```

### 3. Embedding Models
Convert text to vectors.

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Using OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Using open-source (no API key needed)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

### 4. Vector Stores
Store and retrieve vectors.

```python
from langchain.vectorstores import FAISS, Pinecone, Chroma

# Store embeddings locally with FAISS
vector_store = FAISS.from_documents(chunks, embeddings)

# Save for later use
vector_store.save_local("faiss_index")

# Load from disk
vector_store = FAISS.load_local("faiss_index", embeddings)
```

### 5. Retrievers
Query vector store to get relevant documents.

```python
# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Return top 5 documents
)

# Retrieve relevant documents
relevant_docs = retriever.get_relevant_documents("Your query here")
```

### 6. LLM
Generate responses using retrieved context.

```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Using OpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Response
response = llm.predict("Your prompt here")
```

---

## RAG Workflow

### Step 1: Preparation Phase (One-time)
```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# 1. Load documents
loader = TextLoader("knowledge.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Store in vector database
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("knowledge_base")

print(f"Stored {len(chunks)} chunks in vector database")
```

### Step 2: Query Phase (Repeated)
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# 1. Load vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("knowledge_base", embeddings)

# 2. Retrieve relevant documents
query = "What is machine learning?"
relevant_docs = vector_store.similarity_search(query, k=3)

# 3. Format context
context = "\n".join([doc.page_content for doc in relevant_docs])

# 4. Create prompt
prompt_template = """
Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 5. Generate answer
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
formatted_prompt = prompt.format(context=context, question=query)
answer = llm.predict(formatted_prompt)

print(answer)
```

---

## Popular Libraries & Tools

### 1. LangChain
Comprehensive framework for LLM applications.

**Installation:**
```bash
pip install langchain openai
```

**Key features:**
- Document loaders
- Text splitters
- Embeddings integration
- Vector store support
- Chains and agents

### 2. LlamaIndex
Specialized for RAG and data indexing.

**Installation:**
```bash
pip install llama-index openai
```

**Key features:**
- Multiple index types
- Automatic document parsing
- Query optimization
- Agent support

### 3. FAISS
Facebook AI Similarity Search (vector database).

**Installation:**
```bash
pip install faiss-cpu
# or for GPU
pip install faiss-gpu
```

### 4. ChromaDB
Lightweight vector database, great for getting started.

**Installation:**
```bash
pip install chromadb
```

### 5. Pinecone
Cloud-based vector database.

**Installation:**
```bash
pip install pinecone-client
```

### 6. Hugging Face Transformers
Pre-trained models and embeddings.

**Installation:**
```bash
pip install transformers sentence-transformers
```

---

## Implementation Examples

### Example 1: Simple RAG Pipeline with LangChain

```python
"""
Simple RAG pipeline: Load documents, create embeddings, answer questions
"""

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class SimpleRAG:
    def __init__(self, doc_path: str, model_name: str = "gpt-3.5-turbo"):
        """Initialize RAG system with documents."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.vector_store = self._create_vector_store(doc_path)
        
    def _create_vector_store(self, doc_path: str):
        """Load documents and create vector store."""
        # Load document
        loader = TextLoader(doc_path)
        documents = loader.load()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        print(f"Created vector store with {len(chunks)} chunks")
        return vector_store
    
    def query(self, question: str, num_sources: int = 3) -> str:
        """Answer a question using RAG."""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(
            question, 
            k=num_sources
        )
        
        # Format context
        context = "\n\n".join([
            f"[Source {i+1}]\n{doc.page_content}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        # Create prompt
        prompt_template = """
You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Provide a clear, concise answer. If the answer is not in the context, say so.
"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=question)
        
        return answer

# Usage
if __name__ == "__main__":
    # Initialize RAG
    rag = SimpleRAG("documents/knowledge.txt")
    
    # Ask questions
    questions = [
        "What is Python?",
        "How do I use RAG?",
        "Explain embeddings"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        answer = rag.query(q)
        print(f"A: {answer}")
```

### Example 2: RAG with Multiple Document Types

```python
"""
RAG pipeline supporting PDF, TXT, and web content
"""

from langchain.document_loaders import (
    PDFLoader, 
    TextLoader, 
    UnstructuredURLLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from langchain.schema import Document

class MultiSourceRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_store = None
        
    def load_documents(self, sources: dict) -> List[Document]:
        """Load documents from multiple sources."""
        all_documents = []
        
        # Load PDFs
        if "pdfs" in sources:
            for pdf_path in sources["pdfs"]:
                loader = PDFLoader(pdf_path)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"Loaded {len(docs)} pages from {pdf_path}")
        
        # Load text files
        if "texts" in sources:
            for txt_path in sources["texts"]:
                loader = TextLoader(txt_path)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"Loaded document from {txt_path}")
        
        # Load from directory
        if "directory" in sources:
            loader = DirectoryLoader(
                sources["directory"],
                glob="**/*.txt"
            )
            docs = loader.load()
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} documents from directory")
        
        # Load URLs
        if "urls" in sources:
            loader = UnstructuredURLLoader(urls=sources["urls"])
            docs = loader.load()
            all_documents.extend(docs)
            print(f"Loaded {len(docs)} documents from URLs")
        
        return all_documents
    
    def create_vector_store(self, documents: List[Document]):
        """Create vector store from documents."""
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print(f"Created vector store with {len(chunks)} chunks")
        
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)

# Usage
if __name__ == "__main__":
    rag = MultiSourceRAG()
    
    # Load from multiple sources
    sources = {
        "pdfs": ["docs/paper1.pdf", "docs/paper2.pdf"],
        "texts": ["docs/readme.txt"],
        "directory": "knowledge_base/",
        "urls": ["https://example.com/article"]
    }
    
    documents = rag.load_documents(sources)
    rag.create_vector_store(documents)
    
    # Search
    results = rag.search("What is machine learning?")
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(doc.page_content[:200] + "...")
```

---

## LangChain RAG Example

### Complete End-to-End RAG with LangChain

```python
"""
Complete RAG system using LangChain
"""

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load Documents
print("1. Loading documents...")
loader = DirectoryLoader(
    "knowledge_base/",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# 2. Split Documents
print("\n2. Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Create Embeddings
print("\n3. Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # or "cuda" for GPU
)

# 4. Create Vector Store
print("\n4. Creating vector store...")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store created")

# 5. Create Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 6. Create LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# 7. Create QA Chain
prompt_template = """
You are an expert AI assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Provide a helpful, detailed answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 8. Use the System
print("\n" + "="*50)
print("RAG System Ready!")
print("="*50)

queries = [
    "What is Python?",
    "How does RAG work?",
    "Explain machine learning"
]

for query in queries:
    print(f"\nQuestion: {query}")
    result = qa_chain({"query": query})
    
    print(f"\nAnswer:\n{result['result']}")
    print(f"\nSources:")
    for i, doc in enumerate(result['source_documents']):
        print(f"  [{i+1}] {doc.metadata.get('source', 'Unknown')}")
```

---

## LlamaIndex RAG Example

### Complete RAG using LlamaIndex

```python
"""
RAG system using LlamaIndex (formerly GPT Index)
"""

from llama_index import (
    GPTVectorStoreIndex, 
    SimpleDirectoryReader, 
    ServiceContext,
    StorageContext
)
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import SimpleVectorStore
from llama_index.llms import OpenAI

# 1. Load Documents
print("1. Loading documents...")
documents = SimpleDirectoryReader("knowledge_base/").load_data()
print(f"Loaded {len(documents)} documents")

# 2. Set up Service Context
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4", temperature=0.7),
    embed_model="local:BAAI/bge-small-en-v1.5"
)

# 3. Create Index
print("\n2. Creating index...")
index = GPTVectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)
print("Index created")

# 4. Create Query Engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)

# 5. Ask Questions
print("\n" + "="*50)
print("LlamaIndex RAG System Ready!")
print("="*50)

queries = [
    "What is Python?",
    "How does RAG work?",
    "Explain machine learning"
]

for query in queries:
    print(f"\nQuestion: {query}")
    response = query_engine.query(query)
    
    print(f"\nAnswer:\n{response}")
    print(f"\nConfidence: High" if response.source_nodes else "Low")
```

---

## Vector Databases

### FAISS (Local, Free)

**Installation:**
```bash
pip install faiss-cpu
```

**Usage:**
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Create and save
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")

# Load
vector_store = FAISS.load_local("faiss_index", embeddings)

# Search
results = vector_store.similarity_search("query", k=5)
```

### ChromaDB (Lightweight)

**Installation:**
```bash
pip install chromadb
```

**Usage:**
```python
from langchain.vectorstores import Chroma

# Create
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

# Load
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Search
results = vector_store.similarity_search("query", k=5)
```

### Pinecone (Cloud-based)

**Installation:**
```bash
pip install pinecone-client
```

**Usage:**
```python
import pinecone
from langchain.vectorstores import Pinecone

# Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create index
pinecone.create_index("index-name", dimension=384)

# Upsert vectors
vector_store = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="index-name"
)

# Search
results = vector_store.similarity_search("query", k=5)
```

---

## Embeddings

### OpenAI Embeddings (Most Accurate but Paid)

```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # or text-embedding-3-large
    api_key="YOUR_API_KEY"
)

# Get embedding for a text
embedding = embeddings.embed_query("Hello world")
print(f"Embedding dimension: {len(embedding)}")
```

### HuggingFace Embeddings (Free, Local)

```python
from langchain.embeddings import HuggingFaceEmbeddings

# Small model (fast)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Larger model (better quality)
embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2"
)

# Get embedding
embedding = embeddings.embed_query("Hello world")
```

### Sentence Transformers (Direct Usage)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed single text
embedding = model.encode("Hello world")

# Embed multiple texts
embeddings = model.encode([
    "This is a sample sentence",
    "Each sentence is converted to a vector"
])

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embedding], embeddings)
```

---

## Best Practices

### 1. Document Preparation
```python
# Clean and preprocess documents
def preprocess_documents(documents):
    """Clean documents before creating embeddings."""
    for doc in documents:
        # Remove extra whitespace
        doc.page_content = " ".join(doc.page_content.split())
        
        # Remove special characters
        doc.page_content = doc.page_content.replace("\\n", "\n")
        
        # Add metadata
        if "source" not in doc.metadata:
            doc.metadata["source"] = "unknown"
    
    return documents
```

### 2. Chunk Optimization
```python
# Choose chunk size based on your LLM's context window
# Longer context window → larger chunks possible

from langchain.text_splitter import RecursiveCharacterTextSplitter

# For GPT-4 (128k context)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,    # Larger chunks
    chunk_overlap=400
)

# For GPT-3.5 (4k context)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Smaller chunks
    chunk_overlap=200
)
```

### 3. Reranking for Better Results
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Retrieve more documents initially
base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Then rerank
compressor = CohereRerank(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use reranked retriever
results = compression_retriever.get_relevant_documents(query)
```

### 4. Caching Embeddings
```python
# Cache embeddings to avoid recomputing
import os
import pickle

def save_vector_store(vector_store, path):
    """Save vector store to disk."""
    vector_store.save_local(path)
    print(f"Vector store saved to {path}")

def load_vector_store(path, embeddings):
    """Load vector store from disk."""
    from langchain.vectorstores import FAISS
    return FAISS.load_local(path, embeddings)

# Usage
if os.path.exists("vector_store"):
    vector_store = load_vector_store("vector_store", embeddings)
else:
    vector_store = FAISS.from_documents(chunks, embeddings)
    save_vector_store(vector_store, "vector_store")
```

### 5. Error Handling
```python
try:
    results = vector_store.similarity_search(query, k=5)
    
    if not results:
        print("No relevant documents found")
        return "I couldn't find information relevant to your query"
    
    # Process results
    answer = generate_answer(results, query)
    return answer
    
except Exception as e:
    print(f"Error during search: {e}")
    return "An error occurred while processing your question"
```

---

## Performance Optimization

### 1. Batch Processing
```python
# Process multiple queries efficiently
queries = [
    "What is Python?",
    "How does ML work?",
    "Explain deep learning"
]

# Batch retrieve
all_results = []
for query in queries:
    results = vector_store.similarity_search(query, k=5)
    all_results.extend(results)

# Remove duplicates
unique_results = list(set(all_results))
```

### 2. Lazy Loading
```python
# Load documents incrementally
from langchain.document_loaders import TextLoader

def load_documents_lazy(directory):
    """Load documents lazily to save memory."""
    import os
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            path = os.path.join(directory, filename)
            yield TextLoader(path).load()
```

### 3. Vector Store Optimization
```python
# Use GPU for faster similarity search
import faiss

# CPU index
index = faiss.IndexFlatL2(dimension)

# GPU index (faster)
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
```

### 4. Token Counting
```python
# Estimate costs and optimize prompt length
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text):
    return len(encoding.encode(text))

# Use in RAG
context_tokens = count_tokens(context)
query_tokens = count_tokens(query)
total = context_tokens + query_tokens

print(f"Context tokens: {context_tokens}")
print(f"Query tokens: {query_tokens}")
print(f"Total tokens: {total}")
```

---

## Common Use Cases

### 1. Document Q&A Chatbot
```python
# Answer questions about documents
class DocumentQA:
    def __init__(self, doc_paths):
        self.retriever = self._setup_retriever(doc_paths)
    
    def answer(self, question):
        docs = self.retriever.get_relevant_documents(question)
        return self._generate_answer(docs, question)
```

### 2. Research Assistant
```python
# Help with research using papers/articles
class ResearchAssistant:
    def __init__(self, paper_directory):
        self.papers = self._load_papers(paper_directory)
        self.vector_store = self._create_index(self.papers)
    
    def find_papers_on_topic(self, topic, k=10):
        return self.vector_store.similarity_search(topic, k=k)
    
    def summarize_topic(self, topic):
        relevant_papers = self.find_papers_on_topic(topic)
        # Generate summary from papers
```

### 3. Customer Support Bot
```python
# Answer customer questions using knowledge base
class SupportBot:
    def __init__(self, faq_path, docs_path):
        self.faqs = self._load_faqs(faq_path)
        self.docs = self._load_docs(docs_path)
        self.retriever = self._create_retriever()
    
    def answer_ticket(self, ticket_text):
        # Find relevant FAQ/doc
        relevant = self.retriever.get_relevant_documents(ticket_text)
        # Generate response
```

### 4. Code Documentation Assistant
```python
# Help developers find documentation
class DocAssistant:
    def __init__(self, docs_directory):
        self.docs = self._load_documentation(docs_directory)
        self.index = self._create_index(self.docs)
    
    def find_relevant_section(self, code_snippet):
        return self.index.similarity_search(code_snippet, k=3)
    
    def suggest_examples(self, function_name):
        # Find examples for function
```

---

## Troubleshooting

### Issue: Low Quality Results
**Solutions:**
- Increase chunk overlap
- Use better embedding model (e.g., all-mpnet-base-v2)
- Try reranking
- Increase k (number of retrieved documents)
- Clean and preprocess documents better

### Issue: Slow Queries
**Solutions:**
- Use GPU for embeddings/similarity search
- Reduce chunk size
- Implement caching
- Use batch processing
- Choose faster embedding model

### Issue: Out of Memory
**Solutions:**
- Use streaming/lazy loading
- Reduce batch size
- Use smaller embedding model
- Implement pagination
- Use cloud vector database instead of local

### Issue: Hallucinations Still Occurring
**Solutions:**
- Better prompt engineering
- Increase number of retrieved documents
- Use source verification in prompts
- Implement confidence scoring
- Use smaller, more focused documents

---

## Resources

- **LangChain Documentation:** https://python.langchain.com
- **LlamaIndex Docs:** https://docs.llamaindex.ai
- **FAISS Guide:** https://github.com/facebookresearch/faiss
- **Hugging Face Models:** https://huggingface.co/models
- **RAG Survey:** https://arxiv.org/abs/2312.10997
- **Vector DB Comparison:** https://github.com/erikbern/ann-benchmarks

---

**Last Updated:** January 2026
**Difficulty Level:** Intermediate to Advanced