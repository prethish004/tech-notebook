Below is a **complete technical Markdown guide** including:

1️⃣ **CrewAI Architecture Diagram**
2️⃣ **Full list of 40+ CrewAI tools categorized**

Based on the official docs of **CrewAI** and the CrewAI tools repository. ([CrewAI Documentation][1])

You can save this directly as **`crewai_architecture_and_tools.md`**.

---

# CrewAI Complete Architecture & Tools Guide

# 1. CrewAI Architecture Diagram

CrewAI is a **multi-agent orchestration framework** designed for building collaborative AI systems where multiple agents work together using tools, memory, and workflows. ([CrewAI Documentation][2])

---

## High-Level Architecture

```text
                         USER INPUT
                              │
                              ▼
                     ┌─────────────────┐
                     │     CrewAI      │
                     │ Orchestration   │
                     └────────┬────────┘
                              │
                              ▼
                        ┌──────────┐
                        │   Crew   │
                        │ Team of  │
                        │ Agents   │
                        └─────┬────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼

      ┌──────────┐     ┌──────────┐     ┌──────────┐
      │  Agent 1 │     │  Agent 2 │     │  Agent 3 │
      │Researcher│     │  Writer  │     │ Analyst  │
      └─────┬────┘     └─────┬────┘     └─────┬────┘
            │                │                │
            ▼                ▼                ▼
        ┌────────┐      ┌────────┐      ┌────────┐
        │  Task  │      │  Task  │      │  Task  │
        └────┬───┘      └────┬───┘      └────┬───┘
             │               │               │
             ▼               ▼               ▼
         ┌─────────────────────────────────────┐
         │              Tools                  │
         │  Search • File • Code • APIs        │
         └──────────────┬──────────────────────┘
                        │
                        ▼
                 External Systems
         (Web, Databases, APIs, Files)

                        │
                        ▼
                  Final Output
```

---

# CrewAI Internal Architecture

```text
CrewAI Framework
│
├── Agents
│     ├ Role
│     ├ Goal
│     ├ Memory
│     └ Tools
│
├── Tasks
│     ├ Description
│     ├ Expected Output
│     └ Assigned Agent
│
├── Crew
│     ├ Agents
│     ├ Tasks
│     └ Process Type
│
├── Process Types
│     ├ Sequential
│     ├ Hierarchical
│     └ Hybrid
│
├── Tools
│     ├ Web Search
│     ├ File Processing
│     ├ APIs
│     ├ Databases
│     └ AI Models
│
└── Flows
      ├ Event Triggers
      ├ Routing
      ├ State Management
      └ Workflow Automation
```

---

# CrewAI Runtime Execution Flow

```text
User Prompt
     │
     ▼
Crew Manager
     │
     ▼
Task Planning
     │
     ▼
Agent Assignment
     │
     ▼
Agent Execution
     │
     ▼
Tool Usage
     │
     ▼
External Data Retrieval
     │
     ▼
LLM Processing
     │
     ▼
Final Response
```

---

# 2. Full List of CrewAI Tools (40+)

CrewAI provides **40+ prebuilt tools** for agents to interact with external systems like websites, files, APIs, and databases. ([CrewAI Documentation][1])

---

# Search & Research Tools

| Tool                | Purpose                    |
| ------------------- | -------------------------- |
| SerperDevTool       | Google search API          |
| EXASearchTool       | Advanced AI web search     |
| BraveSearchTool     | Brave search API           |
| FirecrawlSearchTool | Search websites            |
| GithubSearchTool    | Search GitHub repositories |
| WebsiteSearchTool   | Search inside websites     |
| ArxivPaperTool      | Search research papers     |

---

# Web Scraping & Crawling Tools

| Tool                         | Purpose                      |
| ---------------------------- | ---------------------------- |
| ScrapeWebsiteTool            | Scrape entire website        |
| ScrapeElementFromWebsiteTool | Extract specific elements    |
| FirecrawlCrawlWebsiteTool    | Crawl websites               |
| FirecrawlScrapeWebsiteTool   | Web scraping using Firecrawl |
| BrowserbaseLoadTool          | Browser automation           |
| SeleniumScrapingTool         | Selenium browser scraping    |
| ApifyActorsTool              | Run Apify scraping actors    |

---

# File & Document Tools

| Tool                | Purpose                |
| ------------------- | ---------------------- |
| FileReadTool        | Read files             |
| FileWriteTool       | Write files            |
| DirectoryReadTool   | Read directory content |
| DirectorySearchTool | Search directories     |
| TXTSearchTool       | Search text files      |
| CSVSearchTool       | Search CSV data        |
| JSONSearchTool      | Search JSON files      |
| XMLSearchTool       | Search XML files       |
| DOCXSearchTool      | Search Word documents  |
| PDFSearchTool       | Search PDFs            |
| MDXSearchTool       | Search Markdown files  |

---

# Database Tools

| Tool                     | Purpose            |
| ------------------------ | ------------------ |
| PGSearchTool             | PostgreSQL search  |
| MySQLSearchTool          | MySQL queries      |
| MongoDBVectorSearchTool  | MongoDB vector DB  |
| QdrantVectorSearchTool   | Qdrant vector DB   |
| WeaviateVectorSearchTool | Weaviate vector DB |

---

# AI / ML Tools

| Tool                | Purpose             |
| ------------------- | ------------------- |
| CodeInterpreterTool | Execute Python code |
| VisionTool          | Image understanding |
| DallETool           | Image generation    |
| StagehandTool       | AI automation tool  |

---

# RAG (Retrieval Augmented Generation)

| Tool               | Purpose                   |
| ------------------ | ------------------------- |
| RagTool            | Generic RAG pipeline      |
| CodeDocsSearchTool | Search code documentation |
| LlamaIndexTool     | Use LlamaIndex pipelines  |

---

# Data Platform Tools

| Tool                      | Purpose                     |
| ------------------------- | --------------------------- |
| BrightDataSearchTool      | BrightData web search       |
| BrightDataDatasetTool     | Dataset retrieval           |
| BrightDataWebUnlockerTool | Bypass anti-bot protections |

---

# Integration Tools

| Tool            | Purpose                 |
| --------------- | ----------------------- |
| ComposioTool    | Integrate SaaS APIs     |
| EmailTool       | Send emails             |
| SlackTool       | Slack automation        |
| GoogleDriveTool | File storage automation |

---

# Example Tool Usage

```python
from crewai_tools import SerperDevTool
from crewai import Agent

search_tool = SerperDevTool()

agent = Agent(
    role="Researcher",
    goal="Find latest AI news",
    tools=[search_tool],
    verbose=True
)
```

---

# CrewAI Tool Categories

```text
CrewAI Tools
│
├── Search Tools
│     ├ SerperDevTool
│     ├ EXASearchTool
│     └ BraveSearchTool
│
├── Scraping Tools
│     ├ ScrapeWebsiteTool
│     ├ SeleniumScrapingTool
│     └ FirecrawlTools
│
├── File Tools
│     ├ FileReadTool
│     ├ DirectoryReadTool
│     └ CSVSearchTool
│
├── Database Tools
│     ├ PGSearchTool
│     ├ MySQLSearchTool
│     └ Vector DB Tools
│
├── AI Tools
│     ├ CodeInterpreterTool
│     ├ VisionTool
│     └ DallETool
│
└── Integration Tools
      ├ Slack
      ├ Gmail
      └ APIs
```

---

# Real Example Multi-Agent Workflow

```text
User: "Write a report about AI trends"

CrewAI Flow

Research Agent
     │
     ▼
SerperDevTool (web search)

     │
     ▼
Writer Agent
     │
     ▼
FileWriteTool (save report)

     │
     ▼
Final Output
```

---

# Typical CrewAI Project Structure

```text
crewai_project/
│
├── agents.py
├── tasks.py
├── crew.py
├── tools.py
├── main.py
│
├── data/
│
└── .env
```

---

# Real Use Cases

CrewAI is used for:

* Autonomous AI assistants
* Research automation
* Coding agents
* Customer support AI
* Data analysis pipelines
* Marketing automation

---
/overview?utm_source=chatgpt.com "Tools Overview"
[2]: https://docs.crewai.com/?utm_source=chatgpt.com "CrewAI Documentation - CrewAI"
