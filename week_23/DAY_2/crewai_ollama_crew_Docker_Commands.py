#%% Crew 2

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool

# Use local Ollama model, e.g., 'llama3'
local_llm = LLM(model="ollama/gpt-oss:120b-cloud", base_url="http://localhost:11434")

# Tool to scrape website content
scrape_tool = ScrapeWebsiteTool()

# Agent to scrape the website
scraper_agent = Agent(
    role="Web Scraper",
    goal="Extract raw content from the specified website URL",
    backstory="An expert at navigating websites and collecting data from them.",
    tools=[scrape_tool],
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Agent to analyze and summarize the content
analyzer_agent = Agent(
    role="Content Analyst",
    goal="Analyze the content extracted from a website",
    backstory="A skilled analyst who turns raw data into understandable insights.",
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Task for scraping
scrape_task = Task(
    description="Scrape all the readable content from the website {url}.",
    expected_output="Raw website text content scraped from {url}.",
    agent=scraper_agent,
)

# Task for summarizing
summarize_task = Task(
    description="Take the scraped content from the website and summarize it into a clear, informative report.",
    expected_output="Take content from {url}, make short summary.",
    agent=analyzer_agent,
)

# Task for summarizing
important_commands_task = Task(
    description="Take the scraped content from the website and extract main docker commands.",
    expected_output="Take content from {url}, extract main docker commands withs short information or deteils.",
    agent=analyzer_agent,
)

# Create the crew
crew = Crew(
    agents=[scraper_agent, analyzer_agent],
    tasks=[scrape_task, summarize_task, important_commands_task],
    process=Process.sequential
)

# Kick it off
result = crew.kickoff(inputs={"url": "https://docker-curriculum.com/"})
print(result)
