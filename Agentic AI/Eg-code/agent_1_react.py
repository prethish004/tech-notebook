"""
Why this exists
LLMs are bad at one-shot answers for complex tasks.
ReAct allows:
  Reason step
  Tool usage
  Observation
  Loop until solved

Used in:
  ChatGPT tools
  Search agents
  Debug agents
REACT AGENT (Reason + Act)

Pattern:
Thought → Action → Observation → Thought → Answer

Why:
- Breaks complex problems into steps
- Reduces hallucination

✅ When to use
  ✔ Unknown info
  ✔ Needs tools
  ✔ Multi-step reasoning
"""

def search_tool(query):
    return f"Search result for: {query}"

def react_agent(question):
    print("Thought: I need external knowledge.")
    
    print("Action: Searching...")
    observation = search_tool(question)
    print("Observation:", observation)
    
    print("Thought: I can now answer.")
    answer = f"Based on info: {observation}"
    
    return answer

print(react_agent("Who created Python?"))
