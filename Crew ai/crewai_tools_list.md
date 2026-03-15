---

# 50+ CrewAI Tools (Complete List)

CrewAI tools allow agents to **perform actions outside the LLM**, such as retrieving data, running code, or interacting with APIs. They extend agent capabilities and enable complex automation workflows. ([Leanware][2])

---

# 1. Search & Research Tools

| Tool                    | Purpose                         |
| ----------------------- | ------------------------------- |
| SerperDevTool           | Google search results API       |
| SerperApiTool           | Web search via Serper           |
| EXASearchTool           | AI-powered semantic web search  |
| BraveSearchTool         | Brave search engine API         |
| WebsiteSearchTool       | Search inside websites          |
| GithubSearchTool        | Search GitHub repositories      |
| ArxivPaperTool          | Search academic research papers |
| StackOverflowSearchTool | Search programming Q&A          |
| NewsSearchTool          | Search news articles            |
| WikipediaSearchTool     | Query Wikipedia data            |

---

# 2. Web Scraping & Browser Automation

| Tool                         | Purpose                        |
| ---------------------------- | ------------------------------ |
| ScrapeWebsiteTool            | Scrape full webpage content    |
| ScrapeElementFromWebsiteTool | Extract specific HTML elements |
| FirecrawlSearchTool          | Crawl and search websites      |
| FirecrawlScrapeWebsiteTool   | Web scraping using Firecrawl   |
| FirecrawlCrawlWebsiteTool    | Large-scale site crawling      |
| SeleniumScrapingTool         | Browser-based scraping         |
| BrowserbaseLoadTool          | Browser automation             |
| ApifyActorsTool              | Run Apify scraping actors      |
| PlaywrightBrowserTool        | Playwright browser automation  |
| RequestsWebTool              | HTTP web requests              |

---

# 3. File Management Tools

| Tool                | Purpose                   |
| ------------------- | ------------------------- |
| FileReadTool        | Read files                |
| FileWriteTool       | Write files               |
| DirectoryReadTool   | Read directory structure  |
| DirectorySearchTool | Search files in directory |
| FileDeleteTool      | Delete files              |
| FileMoveTool        | Move files                |
| FileCopyTool        | Copy files                |
| ZipFileTool         | Compress files            |
| UnzipFileTool       | Extract ZIP archives      |

---

# 4. Document Processing Tools

| Tool           | Purpose               |
| -------------- | --------------------- |
| PDFSearchTool  | Search inside PDFs    |
| DOCXSearchTool | Search Word documents |
| TXTSearchTool  | Search text files     |
| MDXSearchTool  | Search Markdown files |
| CSVSearchTool  | Search CSV datasets   |
| JSONSearchTool | Search JSON data      |
| XMLSearchTool  | Search XML documents  |
| HTMLSearchTool | Parse HTML files      |

---

# 5. Database Tools

| Tool                     | Purpose                 |
| ------------------------ | ----------------------- |
| PGSearchTool             | PostgreSQL query tool   |
| MySQLSearchTool          | MySQL database queries  |
| SQLiteSearchTool         | SQLite database queries |
| MongoDBVectorSearchTool  | MongoDB vector search   |
| QdrantVectorSearchTool   | Qdrant vector DB        |
| WeaviateVectorSearchTool | Weaviate vector DB      |
| PineconeVectorSearchTool | Pinecone vector DB      |
| RedisVectorSearchTool    | Redis vector search     |
| ChromaVectorSearchTool   | Chroma DB search        |

---

# 6. AI / ML Tools

| Tool                | Purpose                |
| ------------------- | ---------------------- |
| CodeInterpreterTool | Execute Python code    |
| VisionTool          | Image analysis         |
| DALLETool           | Image generation       |
| SpeechToTextTool    | Convert speech to text |
| TextToSpeechTool    | Convert text to speech |
| EmbeddingTool       | Generate embeddings    |
| StagehandTool       | AI workflow automation |

---

# 7. RAG & Knowledge Tools

| Tool                     | Purpose                        |
| ------------------------ | ------------------------------ |
| RagTool                  | Retrieval-Augmented Generation |
| CodeDocsSearchTool       | Search code documentation      |
| LlamaIndexTool           | LlamaIndex integration         |
| VectorStoreRetrieverTool | Retrieve vector data           |
| KnowledgeBaseTool        | Query knowledge base           |

---

# 8. Data & Analytics Tools

| Tool                  | Purpose             |
| --------------------- | ------------------- |
| PandasAnalysisTool    | Dataframe analysis  |
| SQLQueryTool          | Execute SQL queries |
| DataVisualizationTool | Create charts       |
| DatasetLoaderTool     | Load datasets       |
| DataCleaningTool      | Clean datasets      |

---

# 9. Communication Tools

| Tool                | Purpose               |
| ------------------- | --------------------- |
| EmailTool           | Send emails           |
| SlackTool           | Slack messaging       |
| DiscordTool         | Discord bot messaging |
| TelegramTool        | Telegram integration  |
| SMSNotificationTool | Send SMS alerts       |

---

# 10. Enterprise SaaS Integration Tools

CrewAI enterprise edition provides connectors for common business systems. ([CrewAI Documentation][3])

| Tool            | Purpose                 |
| --------------- | ----------------------- |
| GmailTool       | Email management        |
| GoogleDriveTool | Cloud storage           |
| NotionTool      | Notion workspace        |
| JiraTool        | Issue tracking          |
| ClickUpTool     | Task management         |
| AsanaTool       | Project management      |
| LinearTool      | Developer task tracking |
| GithubTool      | Repo management         |
| SalesforceTool  | CRM integration         |
| HubSpotTool     | CRM automation          |

---

# 11. DevOps & Infrastructure Tools

| Tool              | Purpose              |
| ----------------- | -------------------- |
| DockerTool        | Container operations |
| KubernetesTool    | Manage Kubernetes    |
| AWSResourceTool   | AWS services         |
| AzureResourceTool | Azure services       |
| GCPResourceTool   | Google Cloud         |

---

# 12. Security & Compliance Tools

| Tool                | Purpose                   |
| ------------------- | ------------------------- |
| CredentialVaultTool | Secure credential access  |
| AuditLogTool        | Security logging          |
| AccessControlTool   | Role-based access control |

---

# Example Using Multiple Tools

```python
from crewai import Agent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileWriteTool

search_tool = SerperDevTool()
scraper = ScrapeWebsiteTool()
writer = FileWriteTool()

agent = Agent(
    role="AI Researcher",
    goal="Collect and store AI research information",
    tools=[search_tool, scraper, writer],
    verbose=True
)
```

---

# Typical Tool Workflow

```
User Request
     │
     ▼
Agent decides tool
     │
     ▼
Search Tool → find info
     │
     ▼
Scraper Tool → extract data
     │
     ▼
File Tool → store result
```

---

# Key Features of CrewAI Tools

* Modular architecture
* Compatible with Python libraries
* Async execution support
* Caching & error handling
* Supports custom tool creation

These tools enable agents to interact with **websites, APIs, files, databases, and enterprise systems** for complex automation tasks. ([Leanware][2])

---

[1]: https://docs.crewai.com/en/tools/overview?utm_source=chatgpt.com "Tools Overview"
[2]: https://www.leanware.co/insights/crewai-tools-guide?utm_source=chatgpt.com "CrewAI Tools - Guide, Installation & Popular Modules"
[3]: https://docs.crewai.com/en/enterprise/features/tools-and-integrations?utm_source=chatgpt.com "Tools & Integrations"
