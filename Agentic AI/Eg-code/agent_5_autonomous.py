5️⃣ GOAL-DRIVEN AUTONOMOUS AGENT (Mini AutoGPT)
"""
Why this exists
Instead of answering, the agent acts until the goal is met.
This is true Agentic AI.
AUTONOMOUS GOAL-DRIVEN AGENT

Why:
- Self-directed execution
- Multi-step autonomy

✅ When to use
✔ Automation
✔ Research agents
✔ Task bots
"""

def perceive(goal):
    return "Missing information"

def act(state):
    return "Gather information"

def reflect(state):
    return "Enough information"

def autonomous_agent(goal):
    state = "start"
    
    while state != "done":
        print("Perception:", perceive(goal))
        print("Action:", act(state))
        state = "done"
        
    return "Goal achieved!"

print(autonomous_agent("Write AI report"))
