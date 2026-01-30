"""
FULL BASED AGENTIC AI SYSTEM (FROM SCRATCH)

This agent demonstrates:
- Goal-driven planning
- ReAct reasoning loop
- Tool calling
- Memory usage
- Controlled execution

🧠 What This Agent Can Do (Capabilities)

✅ Goal-driven execution
✅ Planner → Executor separation
✅ Tool calling (Search / Calculator)
✅ Memory (short-term + task memory)
✅ Reason → Act → Observe loop
✅ Stop condition (no infinite loop)
✅ Clear reasoning logs

🏗 Agent Architecture (Mental Model)
GOAL
 ↓
PLAN (steps)
 ↓
REACT LOOP
 ├── THINK
 ├── DECIDE
 ├── USE TOOL (optional)
 ├── OBSERVE
 └── UPDATE MEMORY
 ↓
FINAL ANSWER


This is true Agentic AI.
Why this matters:
This is the foundation behind AutoGPT, CrewAI, LangGraph
"""

# ==============================
# 🔧 TOOLS
# ==============================

def search_tool(query: str) -> str:
    """Simulates external search"""
    knowledge = {
        "rag": "RAG stands for Retrieval Augmented Generation.",
        "python": "Python was created by Guido van Rossum.",
        "agentic ai": "Agentic AI refers to AI systems that can plan, act, and iterate toward a goal."
    }
    for key in knowledge:
        if key in query.lower():
            return knowledge[key]
    return "No relevant information found."

def calculator_tool(expression: str) -> str:
    """Simple calculator tool"""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid calculation."

# ==============================
# 🧠 MEMORY
# ==============================

class AgentMemory:
    """
    Stores:
    - Observations
    - Tool results
    - Completed steps
    """
    def __init__(self):
        self.entries = []

    def add(self, item: str):
        self.entries.append(item)

    def recall(self) -> str:
        return "\n".join(self.entries[-5:])  # last 5 events

# ==============================
# 🗺 PLANNER
# ==============================

def planner(goal: str):
    """
    Why planner exists:
    - Large goals fail in one shot
    - Breaking tasks improves reliability
    """
    if "what" in goal.lower() or "explain" in goal.lower():
        return [
            "Understand the question",
            "Search for factual information",
            "Compose final explanation"
        ]
    elif "calculate" in goal.lower():
        return [
            "Extract mathematical expression",
            "Compute result",
            "Return answer"
        ]
    else:
        return [
            "Analyze the goal",
            "Gather required info",
            "Generate answer"
        ]

# ==============================
# 🧩 EXECUTOR (REACT LOOP)
# ==============================

def executor(step: str, goal: str, memory: AgentMemory):
    """
    ReAct Pattern:
    Thought → Action → Observation
    """

    print(f"\n🧠 THOUGHT: I need to '{step}'")

    # Decide tool usage
    if "search" in step.lower() or "information" in step.lower():
        print("🔧 ACTION: Using Search Tool")
        observation = search_tool(goal)

    elif "compute" in step.lower() or "calculate" in step.lower():
        print("🔧 ACTION: Using Calculator Tool")
        observation = calculator_tool(goal)

    else:
        print("🧠 ACTION: Reasoning internally")
        observation = "Reasoning completed."

    print("👀 OBSERVATION:", observation)
    memory.add(observation)

# ==============================
# 🤖 AGENT CONTROLLER
# ==============================

def agentic_ai(goal: str):
    """
    Main agent loop
    """
    print("\n==============================")
    print("🎯 GOAL:", goal)
    print("==============================")

    memory = AgentMemory()

    # Step 1: Plan
    plan = planner(goal)
    print("\n🗺 PLAN:")
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")

    # Step 2: Execute plan
    for step in plan:
        executor(step, goal, memory)

    # Step 3: Final answer synthesis
    print("\n🧠 FINAL REASONING:")
    context = memory.recall()
    print(context)

    print("\n✅ FINAL ANSWER:")
    if "rag" in goal.lower():
        print("RAG is a technique that improves LLM accuracy by retrieving external knowledge.")
    elif "python" in goal.lower():
        print("Python was created by Guido van Rossum.")
    else:
        print("Goal processed successfully.")

# ==============================
# ▶ RUN
# ==============================

if __name__ == "__main__":
    agentic_ai("Explain Agentic AI and RAG")
  
  """
  ▶ How to Run
python agentic_ai_base.py

🧪 Example Output (Simplified)
🎯 GOAL: Explain Agentic AI and RAG

🗺 PLAN:
1. Understand the question
2. Search for factual information
3. Compose final explanation

🧠 THOUGHT: I need to 'Search for factual information'
🔧 ACTION: Using Search Tool
👀 OBSERVATION: Agentic AI refers to AI systems that can plan, act...

✅ FINAL ANSWER:
RAG is a technique that improves LLM accuracy by retrieving external knowledge.

🧠 WHY EACH PART EXISTS (VERY IMPORTANT)
Component	Why we need it
Planner	Avoids one-shot failures
Tools	Avoid hallucination
Memory	Enables context & learning
ReAct Loop	Multi-step reasoning
Stop Condition	Prevents infinite loops
🔑 CRITICAL INSIGHT (REMEMBER THIS)

RAG answers questions
AGENTS decide actions

When you combine them → Agentic RAG 🔥
"""
