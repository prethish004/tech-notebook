Perfect! Here's a complete **CrewAI guide** for your `tech-notebook` with all major types + **small, runnable code examples**. You can create `crewai/README.md` with this.

## CrewAI Quickstart Guide

**CrewAI** is a framework for **multi-agent AI teams** where agents have **roles, goals, backstories**, use **tools**, and collaborate via **processes** (sequential, hierarchical, etc.). [datacamp](https://www.datacamp.com/tutorial/crew-ai)

***

## Types of CrewAI Processes

| Process Type | How it works | When to use |
|--------------|--------------|-------------|
| **Sequential** | Agents execute tasks one after another | Linear workflows (step 1 → step 2 → step 3) |
| **Hierarchical** | Manager agent routes tasks to specialists | Complex projects needing coordination [docs.crewai](https://docs.crewai.com/en/learn/hierarchical-process) |
| **Consensual** | Agents discuss and agree before finalizing | Decision-making teams [crewai.mintlify](https://crewai.mintlify.app/en/concepts/agents) |

***

## Installation & Setup

```bash
pip install crewai crewai-tools langchain-openai
```

Set your API key:
```bash
export OPENAI_API_KEY=your_key_here
```

***

## 1. Sequential Crew (Simplest)

**Example:** Research → Write → Review article.

```python
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Agents
researcher = Agent(
    role="Senior Researcher",
    goal="Find accurate, up-to-date information",
    backstory="Expert with access to search tools",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging, accurate articles",
    backstory="Award-winning journalist",
    llm=llm,
    verbose=True
)

reviewer = Agent(
    role="Editor",
    goal="Ensure quality and accuracy",
    backstory="30+ years editing experience",
    llm=llm,
    verbose=True
)

# Tasks
research_task = Task(
    description="Research 'CrewAI best practices' and list top 5 tips.",
    agent=researcher
)

write_task = Task(
    description="Write a 300-word article on CrewAI best practices using {research_task}",
    agent=writer,
    context=[research_task]
)

review_task = Task(
    description="Review and polish the article from {write_task}",
    agent=reviewer,
    context=[write_task]
)

# Crew
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, write_task, review_task],
    process=Process.sequential,
    verbose=2
)

result = crew.kickoff()
print(result)
```


***

## 2. Hierarchical Crew (Manager + Workers)

**Example:** Project manager coordinates developer + QA.

```python
from crewai import Agent, Task, Crew, Process

# Manager
manager = Agent(
    role="Project Manager",
    goal="Coordinate team to deliver high-quality code",
    backstory="Experienced PM who understands tech",
    llm=llm,
    verbose=True
)

developer = Agent(role="Senior Developer", goal="Write clean code", llm=llm, verbose=True)
qa = Agent(role="QA Engineer", goal="Find bugs", llm=llm, verbose=True)

code_task = Task(description="Write a Python function to calculate fibonacci", agent=developer)
review_task = Task(description="Review the code for bugs", agent=qa)

crew = Crew(
    agents=[manager, developer, qa],
    tasks=[code_task, review_task],
    process=Process.hierarchical,  # Manager routes tasks
    manager_agent=manager,  # Required for hierarchical
    verbose=2
)

result = crew.kickoff()
```


***

## 3. Crew with Tools

**Example:** Researcher uses web search tool.

```python
from crewai_tools import SerperDevTool  # pip install crewai-tools

search_tool = SerperDevTool()  # needs SERPER_API_KEY

researcher = Agent(
    role="Market Researcher",
    goal="Research using web search",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

research_task = Task(
    description="Research latest trends in AI agents using web search",
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[research_task], verbose=2)
result = crew.kickoff()
```


***

## 4. Consensual Process

Agents **discuss and vote** before finalizing.

```python
crew = Crew(
    agents=[analyst1, analyst2, decision_maker],
    tasks=[analyze_task1, analyze_task2, decide_task],
    process=Process.consensual,  # They discuss
    verbose=2
)
```


***

## 5. YAML Configuration (Production-style)

Create `agents.yaml`, `tasks.yaml`, then:

```python
from crewai import Crew

crew = Crew.from_yaml("agents.yaml", "tasks.yaml")
result = crew.kickoff()
```


***

## Quick Start Tips

1. **Start sequential** → add tools → try hierarchical.
2. **Use gpt-4o-mini** for cost/speed balance.
3. **Add `verbose=2`** to see agent thinking.
4. **Tools need API keys** (Serper, DuckDuckGo, etc.). [github](https://github.com/crewAIInc/crewAI-examples)
5. **Great for:** research pipelines, code review, content creation, trip planning. [projectpro](https://www.projectpro.io/article/crew-ai-projects-ideas-and-examples/1117)
