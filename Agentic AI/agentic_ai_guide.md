# Agentic AI: Building Autonomous AI Agents

## Table of Contents
1. [What is Agentic AI?](#what-is-agentic-ai)
2. [Core Concepts](#core-concepts)
3. [Agent Architecture](#agent-architecture)
4. [Types of Agents](#types-of-agents)
5. [Agent Loop & Decision Making](#agent-loop--decision-making)
6. [Tools & Actions](#tools--actions)
7. [Memory Systems](#memory-systems)
8. [Popular Frameworks](#popular-frameworks)
9. [LangChain Agent Example](#langchain-agent-example)
10. [Building Custom Agents](#building-custom-agents)
11. [Multi-Agent Systems](#multi-agent-systems)
12. [Agent Communication](#agent-communication)
13. [Best Practices](#best-practices)
14. [Troubleshooting](#troubleshooting)

---

## What is Agentic AI?

**Agentic AI** refers to AI systems that can autonomously decide what actions to take to achieve goals, without being explicitly programmed for every scenario.

### Key Characteristics

- **Autonomy:** Makes decisions independently
- **Goal-Oriented:** Works toward specific objectives
- **Tool Usage:** Uses external tools and APIs
- **Reasoning:** Analyzes situations before acting
- **Adaptability:** Adjusts behavior based on outcomes
- **Persistence:** Continues until goal is achieved

### Agent vs Traditional LLM

| Aspect | Traditional LLM | Agent |
|--------|-----------------|-------|
| **Input** | Prompt → Output | Environment → Decision → Action |
| **Interaction** | Single response | Iterative loops |
| **Tool Use** | Cannot call functions | Can use tools dynamically |
| **Error Handling** | Manual | Automatic recovery |
| **Complexity** | Simple | Complex multi-step tasks |
| **Memory** | Context window only | Long-term memory |

### Real-World Examples

- **Research Agent:** Searches web, reads papers, synthesizes findings
- **Code Agent:** Writes, tests, debugs code autonomously
- **Trading Agent:** Analyzes markets, makes trades, monitors positions
- **Support Agent:** Helps customers, escalates issues, learns patterns
- **Scheduling Agent:** Books meetings, sends reminders, coordinates

---

## Core Concepts

### 1. State
The current condition of the agent and its environment.

```python
state = {
    "goal": "Find cheapest flight to Paris",
    "constraints": {"budget": 500, "dates": "Jan 15-20"},
    "tools_available": ["web_search", "price_compare", "booking"],
    "memory": ["Already searched 3 airlines"],
    "status": "searching"
}
```

### 2. Actions
Discrete steps the agent can take.

```python
actions = [
    {"type": "search", "query": "flights to Paris"},
    {"type": "analyze", "data": "flight_options"},
    {"type": "book", "flight_id": "AA123"},
    {"type": "report", "summary": "Best flight booked"}
]
```

### 3. Observations
Information returned after taking an action.

```python
observation = {
    "action_taken": "search",
    "results": [
        {"airline": "Air France", "price": 450, "rating": 4.5},
        {"airline": "Lufthansa", "price": 480, "rating": 4.2}
    ],
    "status": "success"
}
```

### 4. Reward/Feedback
Signal indicating how well the agent is doing.

```python
feedback = {
    "reward": 0.9,  # 0-1 scale
    "reason": "Found good deal within budget",
    "next_step": "Proceed to booking"
}
```

### 5. Policy
Strategy for deciding which action to take.

```python
# Policy examples:
# 1. Greedy: Always choose highest immediate reward
# 2. Exploration: Try new things to learn
# 3. Cost-benefit: Balance exploration and exploitation
# 4. Heuristic: Use rules based on domain knowledge
```

---

## Agent Architecture

```
┌──────────────────────────────────────────────┐
│         Agent Architecture Flow              │
└──────────────────────────────────────────────┘

    ┌─────────────────────────┐
    │   User Request/Goal     │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Perception Module      │
    │  (Observe Environment)  │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Memory/Context         │
    │  (Retrieve Knowledge)   │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Reasoning/Planning     │
    │  (LLM Decision Making)  │
    └────────────┬────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌──────────────┐      ┌──────────────┐
│ Action Plan  │      │ Tool Calls   │
│ (Next Steps) │      │ (APIs)       │
└──────┬───────┘      └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │  Execute Actions        │
    │  (Call Tools/APIs)      │
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │  Observation            │
    │  (Get Results)          │
    └────────────┬────────────┘
                 │
    ┌────────────┴────────────────┐
    │                             │
    ▼                             ▼
┌──────────────┐      ┌──────────────────┐
│ Update Memory│      │ Check Goal Met?  │
└──────┬───────┘      └──────┬───────────┘
       │                     │
       │            ┌────────┴────────┐
       │            │                 │
       │            ▼                 ▼
       │      Yes [DONE]         No [LOOP]
       │                             │
       └─────────────┬───────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Return Result to User   │
        └──────────────────────────┘
```

---

## Types of Agents

### 1. Reactive Agents
Respond to immediate environment without planning.

```python
class ReactiveAgent:
    """Responds directly to stimuli without planning."""
    
    def perceive(self, observation):
        """React immediately to observation."""
        if "error" in observation:
            return "retry_action"
        elif "success" in observation:
            return "continue"
        else:
            return "wait"
    
    def act(self, action):
        """Execute action immediately."""
        return self.tools[action]()

# Usage
agent = ReactiveAgent()
observation = {"status": "error", "code": 404}
action = agent.perceive(observation)
result = agent.act(action)
```

### 2. Deliberative Agents
Plan before acting using reasoning.

```python
class DeliberativeAgent:
    """Plans actions before execution."""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def plan(self, goal):
        """Generate plan for goal."""
        prompt = f"Goal: {goal}. Create step-by-step plan."
        plan = self.llm.generate(prompt)
        return plan
    
    def execute_plan(self, plan):
        """Execute planned steps."""
        for step in plan:
            result = self.execute_step(step)
            self.memory.append({"step": step, "result": result})
        return self.memory
    
    def execute_step(self, step):
        """Execute single step."""
        tool = self.find_tool(step.tool)
        return tool(step.params)
```

### 3. Learning Agents
Learn from experience and improve over time.

```python
class LearningAgent:
    """Learns and improves from experience."""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.experience_log = []
        self.policy = {}
    
    def record_experience(self, state, action, reward):
        """Record experience for learning."""
        self.experience_log.append({
            "state": state,
            "action": action,
            "reward": reward
        })
    
    def learn(self):
        """Learn from experiences."""
        successful = [e for e in self.experience_log if e["reward"] > 0.5]
        
        for exp in successful:
            state_key = str(exp["state"])
            if state_key not in self.policy:
                self.policy[state_key] = []
            self.policy[state_key].append(exp["action"])
    
    def act(self, state):
        """Act based on learned policy."""
        state_key = str(state)
        if state_key in self.policy:
            return self.policy[state_key][0]  # Use best action
        else:
            return self.choose_action(state)  # Explore
```

### 4. Hierarchical Agents
Organize behavior in levels.

```python
class HierarchicalAgent:
    """Organizes behavior in hierarchical levels."""
    
    def __init__(self):
        self.high_level_goals = []
        self.mid_level_tasks = []
        self.low_level_actions = []
    
    def decompose_goal(self, goal):
        """Break goal into sub-goals."""
        self.high_level_goals.append(goal)
        
        # Decompose
        tasks = self.decompose_to_tasks(goal)
        self.mid_level_tasks.extend(tasks)
        
        # Decompose further
        for task in tasks:
            actions = self.decompose_to_actions(task)
            self.low_level_actions.extend(actions)
    
    def execute(self):
        """Execute hierarchical structure."""
        for action in self.low_level_actions:
            self.execute_action(action)
```

---

## Agent Loop & Decision Making

### The Agent Loop

```python
class BaseAgent:
    """Basic agent loop implementation."""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = 10
        self.history = []
    
    def run(self, goal):
        """Run agent loop until goal achieved."""
        state = {"goal": goal, "step": 0}
        
        for iteration in range(self.max_iterations):
            # 1. THINK - LLM decides next action
            thought = self.think(state)
            
            # 2. PLAN - What action to take
            action_plan = self.plan(thought)
            
            # 3. EXECUTE - Run the action
            observation = self.execute(action_plan)
            
            # 4. REFLECT - Process result
            reflection = self.reflect(observation)
            
            # 5. CHECK - Goal achieved?
            if self.goal_achieved(reflection):
                return {
                    "success": True,
                    "result": reflection,
                    "iterations": iteration + 1
                }
            
            # 6. UPDATE - Update state for next iteration
            state = self.update_state(state, reflection)
            
            # Log history
            self.history.append({
                "iteration": iteration,
                "thought": thought,
                "action": action_plan,
                "observation": observation
            })
        
        return {
            "success": False,
            "result": "Max iterations reached",
            "history": self.history
        }
    
    def think(self, state):
        """LLM thinks about situation."""
        prompt = self.build_prompt(state)
        return self.llm.generate(prompt)
    
    def plan(self, thought):
        """Decide what action to take."""
        if "search" in thought.lower():
            return {"tool": "web_search"}
        elif "calculate" in thought.lower():
            return {"tool": "calculator"}
        else:
            return {"tool": "noop"}
    
    def execute(self, action_plan):
        """Execute the action."""
        tool = self.tools.get(action_plan["tool"])
        if tool:
            return tool.run(action_plan.get("params", {}))
        return None
    
    def reflect(self, observation):
        """Process observation."""
        return {"result": observation}
    
    def goal_achieved(self, reflection):
        """Check if goal is achieved."""
        return reflection.get("success", False)
    
    def update_state(self, state, reflection):
        """Update state for next iteration."""
        state["step"] += 1
        state["last_result"] = reflection
        return state
    
    def build_prompt(self, state):
        """Build prompt for LLM."""
        return f"Goal: {state['goal']}\nStep: {state['step']}"
```

---

## Tools & Actions

### Defining Tools

```python
from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    """Base class for agent tools."""
    
    name: str
    description: str
    
    @abstractmethod
    def run(self, params: dict) -> Any:
        """Execute the tool."""
        pass

class WebSearchTool(Tool):
    """Search the web."""
    
    name = "web_search"
    description = "Search the web for information"
    
    def run(self, params: dict):
        """Execute web search."""
        query = params.get("query", "")
        num_results = params.get("num_results", 5)
        
        # Simulate web search
        results = [
            {"title": "Result 1", "url": "http://...", "snippet": "..."},
            {"title": "Result 2", "url": "http://...", "snippet": "..."}
        ]
        return results

class CalculatorTool(Tool):
    """Perform calculations."""
    
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def run(self, params: dict):
        """Execute calculation."""
        expression = params.get("expression", "")
        try:
            result = eval(expression)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

class FileTool(Tool):
    """Read/write files."""
    
    name = "file_io"
    description = "Read and write files"
    
    def run(self, params: dict):
        """Execute file operation."""
        operation = params.get("operation", "read")
        filepath = params.get("filepath", "")
        
        if operation == "read":
            with open(filepath, 'r') as f:
                content = f.read()
            return {"success": True, "content": content}
        
        elif operation == "write":
            content = params.get("content", "")
            with open(filepath, 'w') as f:
                f.write(content)
            return {"success": True, "message": "File written"}
        
        return {"success": False, "error": "Unknown operation"}

class DatabaseTool(Tool):
    """Query databases."""
    
    name = "database"
    description = "Query and update databases"
    
    def run(self, params: dict):
        """Execute database query."""
        operation = params.get("operation", "query")
        table = params.get("table", "")
        filters = params.get("filters", {})
        
        # Simulate database query
        results = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"}
        ]
        return {"success": True, "results": results}

# Agent with tools
tools = [
    WebSearchTool(),
    CalculatorTool(),
    FileTool(),
    DatabaseTool()
]

agent = BaseAgent(llm=gpt4, tools=tools)
result = agent.run("Calculate 2+2 and search for AI news")
```

---

## Memory Systems

### Short-Term Memory

```python
class ShortTermMemory:
    """Context window memory for current task."""
    
    def __init__(self, max_size=5):
        self.memory = []
        self.max_size = max_size
    
    def add(self, item):
        """Add item to memory."""
        self.memory.append(item)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)  # Remove oldest
    
    def get_all(self):
        """Get all items."""
        return self.memory
    
    def clear(self):
        """Clear memory."""
        self.memory = []
```

### Long-Term Memory

```python
class LongTermMemory:
    """Persistent memory across sessions."""
    
    def __init__(self, db_path="agent_memory.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        """Create necessary tables."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME,
            content TEXT,
            importance FLOAT,
            category TEXT
        )
        """)
        self.conn.commit()
    
    def add(self, content, importance=0.5, category="general"):
        """Add memory."""
        from datetime import datetime
        self.cursor.execute(
            "INSERT INTO memories (timestamp, content, importance, category) VALUES (?, ?, ?, ?)",
            (datetime.now(), content, importance, category)
        )
        self.conn.commit()
    
    def retrieve(self, category=None, top_k=5):
        """Retrieve memories."""
        if category:
            query = "SELECT content FROM memories WHERE category=? ORDER BY importance DESC LIMIT ?"
            self.cursor.execute(query, (category, top_k))
        else:
            query = "SELECT content FROM memories ORDER BY importance DESC LIMIT ?"
            self.cursor.execute(query, (top_k,))
        
        return [row[0] for row in self.cursor.fetchall()]
    
    def update_importance(self, memory_id, importance):
        """Update importance score."""
        self.cursor.execute(
            "UPDATE memories SET importance=? WHERE id=?",
            (importance, memory_id)
        )
        self.conn.commit()
```

### Episodic Memory

```python
class EpisodicMemory:
    """Memory of specific events/interactions."""
    
    def __init__(self):
        self.episodes = []
    
    def record_episode(self, episode):
        """Record a complete episode."""
        episode_data = {
            "timestamp": episode.get("timestamp"),
            "goal": episode.get("goal"),
            "actions": episode.get("actions", []),
            "observations": episode.get("observations", []),
            "outcome": episode.get("outcome"),
            "success": episode.get("success", False)
        }
        self.episodes.append(episode_data)
    
    def get_similar_episodes(self, current_goal, top_k=3):
        """Find similar past episodes."""
        similar = []
        for ep in self.episodes:
            if self._goals_similar(ep["goal"], current_goal):
                similar.append(ep)
        return similar[:top_k]
    
    def _goals_similar(self, goal1, goal2):
        """Check if goals are similar."""
        # Simple string similarity
        return len(set(goal1.split()) & set(goal2.split())) > 0
    
    def learn_from_episode(self, episode):
        """Extract learnings from episode."""
        return {
            "best_actions": [a for a in episode["actions"] if a["successful"]],
            "pitfalls": [a for a in episode["actions"] if not a["successful"]],
            "success_rate": 1.0 if episode["success"] else 0.0
        }
```

---

## Popular Frameworks

### 1. LangChain Agents

**Installation:**
```bash
pip install langchain openai
```

**Features:**
- Pre-built agent types
- Tool integration
- Memory management
- Chain of thought reasoning

### 2. AutoGPT Style

**Key Components:**
- Planning module
- Execution module
- Error handling
- Recursive self-improvement

### 3. ReAct (Reasoning + Acting)

**Combines:**
- Reasoning traces
- Action steps
- Intermediate observations

### 4. OpenAI Function Calling

**Features:**
- Native tool calling
- Structured outputs
- Error handling

---

## LangChain Agent Example

### Complete Working Agent

```python
"""
Complete LangChain agent with tools and memory
"""

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
import requests

# Define tools
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Simulate web search
    results = f"Found articles about: {query}"
    return results

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculation."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(location: str) -> str:
    """Get weather information."""
    # Simulate weather API
    return f"Weather in {location}: 72°F, Sunny"

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {recipient}: {subject}"

# Create LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Create tools list
tools = [web_search, calculator, get_weather, send_email]

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=10
)

# Run agent
if __name__ == "__main__":
    queries = [
        "What is the weather in New York?",
        "Calculate 1234 + 5678",
        "Search for latest AI news",
        "Send me an email reminder about my meeting"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        response = agent.run(query)
        print(f"Agent: {response}")
```

---

## Building Custom Agents

### Agent with Planning

```python
"""
Custom agent with explicit planning phase
"""

class PlanningAgent:
    """Agent that plans before acting."""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.plan = None
        self.execution_log = []
    
    def run(self, goal):
        """Execute agent with planning."""
        print(f"🎯 Goal: {goal}\n")
        
        # Phase 1: Planning
        self.plan = self.create_plan(goal)
        print(f"📋 Plan: {self.plan}\n")
        
        # Phase 2: Execution
        results = self.execute_plan()
        print(f"✅ Results: {results}\n")
        
        return results
    
    def create_plan(self, goal):
        """Create execution plan."""
        prompt = f"""
        Create a step-by-step plan to achieve this goal: {goal}
        
        Available tools: {list(self.tools.keys())}
        
        Format each step as:
        1. [Tool Name]: [Description]
        2. [Tool Name]: [Description]
        ...
        """
        
        plan = self.llm.generate(prompt)
        return plan
    
    def execute_plan(self):
        """Execute plan steps."""
        steps = self.parse_plan(self.plan)
        results = []
        
        for i, step in enumerate(steps, 1):
            tool_name = step.get("tool")
            params = step.get("params", {})
            
            if tool_name in self.tools:
                print(f"Step {i}: Using {tool_name}...")
                result = self.tools[tool_name].run(params)
                results.append(result)
                self.execution_log.append({
                    "step": i,
                    "tool": tool_name,
                    "result": result
                })
                print(f"  Result: {result}\n")
        
        return results
    
    def parse_plan(self, plan_text):
        """Parse plan text into structured steps."""
        # Simple parsing - in production use regex or LLM
        steps = []
        for line in plan_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                steps.append({
                    "tool": parts[0].strip().split('.')[-1].strip(),
                    "description": parts[1].strip()
                })
        return steps

# Usage
agent = PlanningAgent(llm, tools)
result = agent.run("Find the cheapest flight to Paris for Jan 15-20")
```

### Agent with Reflection

```python
"""
Agent that reflects on actions and learns
"""

class ReflectiveAgent:
    """Agent that learns from mistakes."""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.max_retries = 3
        self.lessons_learned = []
    
    def run(self, goal):
        """Run with reflection and retry."""
        attempt = 0
        
        while attempt < self.max_retries:
            attempt += 1
            print(f"\n🔄 Attempt {attempt}/{self.max_retries}")
            
            # Try to achieve goal
            result = self.attempt_goal(goal)
            
            if result["success"]:
                print(f"✅ Success!")
                return result
            
            # Reflect on failure
            reflection = self.reflect(result)
            print(f"🤔 Reflection: {reflection['insight']}")
            
            # Learn lesson
            self.learn(reflection)
        
        return {"success": False, "reason": "Max attempts reached"}
    
    def attempt_goal(self, goal):
        """Try to achieve goal."""
        # Execute action
        result = {"success": False, "error": "Unknown"}
        
        # Return attempt result
        return result
    
    def reflect(self, result):
        """Reflect on what went wrong."""
        error = result.get("error", "Unknown error")
        
        insight = f"Failed because: {error}"
        
        return {
            "error": error,
            "insight": insight,
            "lesson": self.extract_lesson(error)
        }
    
    def learn(self, reflection):
        """Learn from experience."""
        lesson = {
            "error_type": reflection["error"],
            "solution": reflection["lesson"],
            "timestamp": __import__('datetime').datetime.now()
        }
        
        self.lessons_learned.append(lesson)
        print(f"📚 Learned: {lesson['solution']}")
    
    def extract_lesson(self, error):
        """Extract actionable lesson from error."""
        lessons = {
            "timeout": "Try with smaller batch sizes",
            "permission": "Check credentials or permissions",
            "not_found": "Verify resource exists before accessing",
            "rate_limit": "Add delays between requests"
        }
        
        for key, lesson in lessons.items():
            if key in error.lower():
                return lesson
        
        return "Try a different approach"
```

---

## Multi-Agent Systems

### Agent Communication

```python
"""
Multiple agents working together
"""

class MultiAgentSystem:
    """System of cooperating agents."""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = []
    
    def register_agent(self, name, agent):
        """Register an agent."""
        self.agents[name] = agent
        print(f"Registered agent: {name}")
    
    def send_message(self, from_agent, to_agent, message):
        """Send message between agents."""
        self.message_queue.append({
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": __import__('datetime').datetime.now()
        })
    
    def process_messages(self):
        """Process pending messages."""
        while self.message_queue:
            msg = self.message_queue.pop(0)
            to_agent = self.agents.get(msg["to"])
            
            if to_agent:
                response = to_agent.process_message(msg)
                if response:
                    self.send_message(msg["to"], msg["from"], response)
    
    def delegate_task(self, task, primary_agent):
        """Delegate task to appropriate agent."""
        agent = self.agents.get(primary_agent)
        
        if agent.can_handle(task):
            return agent.execute(task)
        else:
            # Find capable agent
            for name, agent in self.agents.items():
                if agent.can_handle(task):
                    self.send_message(primary_agent, name, task)
                    return agent.execute(task)

# Example multi-agent setup
class ResearchAgent:
    """Agent that researches topics."""
    
    def __init__(self):
        self.expertise = ["research", "analysis"]
    
    def can_handle(self, task):
        return any(exp in task.lower() for exp in self.expertise)
    
    def execute(self, task):
        return f"Researching: {task}"

class WritingAgent:
    """Agent that writes content."""
    
    def __init__(self):
        self.expertise = ["write", "compose", "create"]
    
    def can_handle(self, task):
        return any(exp in task.lower() for exp in self.expertise)
    
    def execute(self, task):
        return f"Writing: {task}"

class ReviewAgent:
    """Agent that reviews content."""
    
    def __init__(self):
        self.expertise = ["review", "check", "verify"]
    
    def can_handle(self, task):
        return any(exp in task.lower() for exp in self.expertise)
    
    def execute(self, task):
        return f"Reviewing: {task}"

# Create system
mas = MultiAgentSystem()
mas.register_agent("researcher", ResearchAgent())
mas.register_agent("writer", WritingAgent())
mas.register_agent("reviewer", ReviewAgent())

# Execute tasks
result1 = mas.delegate_task("Research AI trends", "researcher")
result2 = mas.delegate_task("Write an article about AI", "writer")
result3 = mas.delegate_task("Review the article", "reviewer")
```

---

## Agent Communication

### Message Protocol

```python
"""
Structured agent communication protocol
"""

from enum import Enum
from dataclasses import dataclass

class MessageType(Enum):
    """Types of agent messages."""
    REQUEST = "request"
    RESPONSE = "response"
    ACKNOWLEDGE = "acknowledge"
    DELEGATE = "delegate"
    UPDATE = "update"
    ERROR = "error"

@dataclass
class AgentMessage:
    """Structured agent message."""
    
    message_id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: dict
    priority: int = 5  # 1-10
    timestamp: str = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.message_type.value,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp
        }

class MessageBroker:
    """Manages agent communication."""
    
    def __init__(self):
        self.message_queue = []
        self.processed_messages = []
    
    def send(self, message: AgentMessage):
        """Send message."""
        import datetime
        message.timestamp = datetime.datetime.now().isoformat()
        self.message_queue.append(message)
        print(f"Sent: {message.sender} → {message.recipient}")
    
    def receive(self, recipient: str):
        """Receive messages for agent."""
        messages = [m for m in self.message_queue if m.recipient == recipient]
        # Remove from queue
        self.message_queue = [m for m in self.message_queue if m.recipient != recipient]
        return messages
    
    def get_high_priority(self):
        """Get high priority messages."""
        return [m for m in self.message_queue if m.priority >= 8]

# Usage
broker = MessageBroker()

msg = AgentMessage(
    message_id="msg_001",
    sender="agent_a",
    recipient="agent_b",
    message_type=MessageType.REQUEST,
    content={"task": "analyze_data", "data": [1, 2, 3]},
    priority=9
)

broker.send(msg)
messages = broker.receive("agent_b")
```

---

## Best Practices

### 1. Agent Design

```python
# ✅ Good: Clear responsibilities
class DataFetcherAgent:
    """Only responsible for fetching data."""
    def fetch(self, source):
        pass

class DataAnalyzerAgent:
    """Only responsible for analysis."""
    def analyze(self, data):
        pass

# ❌ Bad: Too many responsibilities
class SuperAgent:
    """Does everything poorly."""
    def fetch_and_analyze_and_report(self):
        pass
```

### 2. Error Handling

```python
def safe_tool_execution(tool, params):
    """Execute tool with error handling."""
    try:
        result = tool.run(params)
        return {"success": True, "result": result}
    except ValueError as e:
        return {"success": False, "error": f"Invalid input: {e}"}
    except TimeoutError:
        return {"success": False, "error": "Tool timed out"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
```

### 3. Resource Management

```python
class ResourceAwareAgent:
    """Respects resource constraints."""
    
    def __init__(self, max_api_calls=100, max_memory_mb=500):
        self.max_api_calls = max_api_calls
        self.api_calls_made = 0
        self.max_memory = max_memory_mb
    
    def can_make_api_call(self):
        """Check if can make another API call."""
        return self.api_calls_made < self.max_api_calls
    
    def make_api_call(self, endpoint):
        """Make API call if resources allow."""
        if not self.can_make_api_call():
            return {"error": "API call limit reached"}
        
        self.api_calls_made += 1
        return self.execute_api(endpoint)
```

### 4. Logging & Debugging

```python
import logging

class InstrumentedAgent:
    """Agent with comprehensive logging."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
    def execute_action(self, action):
        """Execute with logging."""
        self.logger.info(f"Executing action: {action}")
        
        try:
            result = self._execute(action)
            self.logger.debug(f"Action result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Action failed: {e}", exc_info=True)
            raise
```

---

## Troubleshooting

### Issue: Agent Gets Stuck in Loop

```python
# Solution: Add iteration limit
if iteration > max_iterations:
    logger.warning("Max iterations reached")
    break

# Also track visited states
visited_states = set()
current_state = get_state()

if current_state in visited_states:
    logger.warning("Detected loop, changing strategy")
    # Change behavior
else:
    visited_states.add(current_state)
```

### Issue: Tool Failures Not Handled

```python
# Solution: Wrap all tool calls
def execute_with_fallback(primary_tool, fallback_tool, params):
    try:
        return primary_tool.run(params)
    except Exception as e:
        logger.warning(f"Primary tool failed: {e}, trying fallback")
        return fallback_tool.run(params)
```

### Issue: Slow Agent Response

```python
# Solution: Add timeouts and parallelization
import asyncio

async def parallel_tool_execution(tools, params_list):
    """Execute multiple tools in parallel."""
    tasks = [
        tool.run_async(params) 
        for tool, params in zip(tools, params_list)
    ]
    return await asyncio.gather(*tasks)
```

### Issue: Agent Makes Poor Decisions

```python
# Solution: Better prompting and examples
system_prompt = """
You are a helpful agent. Before taking any action:
1. Analyze the current situation
2. Consider all available options
3. Weigh pros and cons
4. Choose the best action

Examples of good decisions:
- ...
- ...
"""
```

---

## Resources

- **LangChain Agents:** https://python.langchain.com/docs/modules/agents
- **OpenAI Function Calling:** https://platform.openai.com/docs/guides/function-calling
- **ReAct Paper:** https://arxiv.org/abs/2210.03629
- **AgentGPT:** https://github.com/reworkd/AgentGPT
- **AutoGPT:** https://github.com/Significant-Gravitas/Auto-GPT
- **Agent Benchmarks:** https://github.com/SamuelSchmidgall/AgentBench

---

**Last Updated:** January 2026
**Difficulty Level:** Intermediate to Advanced
