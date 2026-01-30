"""
Why this exists
Large tasks fail if done at once.
So we:
  PLAN first
  EXECUTE step by step
Used heavily in:
  AutoGPT
  LangGraph
  CrewAI

PLANNER–EXECUTOR AGENT

Why:
- Planning improves task success
- Separation of thinking and doing
"""

def planner(goal):
    return [
        "Understand the question",
        "Fetch related information",
        "Compose final answer"
    ]

def executor(step):
    print(f"Executing: {step}")
    return f"Done: {step}"

def planner_executor_agent(goal):
    plan = planner(goal)
    results = []
    
    for step in plan:
        result = executor(step)
        results.append(result)
    
    return "\n".join(results)

print(planner_executor_agent("Explain RAG"))
