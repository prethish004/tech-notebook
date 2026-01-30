"""
MEMORY AGENT

Why this exists
Stateless agents forget everything.
Memory lets agents:
  Learn from past
  Maintain conversation
  Improve decisions
Why:
- Context persistence
- Learning over time

✅ When to use
✔ Chatbots
✔ Assistants
✔ Long tasks
"""

memory = []

def memory_agent(user_input):
    memory.append(user_input)
    
    if len(memory) > 1:
        return f"I remember you said: {memory[-2]}"
    return "Got it!"

print(memory_agent("Hello"))
print(memory_agent("Explain Agentic AI"))
