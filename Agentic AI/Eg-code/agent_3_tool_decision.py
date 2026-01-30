"""
TOOL CALLING AGENT
Why this exists
LLMs shouldn’t always answer directly.
Sometimes:
  Use calculator
  Use DB
  Use API
Why:
- Avoid hallucination
- Delegate tasks to tools
This agent chooses tools.
✅ When to use
✔ Math
✔ DB queries
✔ External APIs
"""

def calculator(a, b):
    return a + b

def knowledge_base(q):
    return "RAG = Retrieval Augmented Generation"

def tool_agent(query):
    if "add" in query:
        return calculator(2, 3)
    elif "rag" in query.lower():
        return knowledge_base(query)
    else:
        return "LLM answer directly"

print(tool_agent("add numbers"))
print(tool_agent("what is rag"))
