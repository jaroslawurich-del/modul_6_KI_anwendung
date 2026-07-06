#%% Crew 1

import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Serper key still needed for the search tool
os.environ["SERPER_API_KEY"] = "44753a64dee969a99a4c86d0a2a077965939a67b"

# Use local Ollama model, e.g., 'llama3'
local_llm = LLM(model="ollama/gpt-oss:120b-cloud", base_url="http://localhost:11434")

# llama3.1:latest 

# Tool to search the web
search_tool = SerperDevTool()

# Agent that performs the search
search_agent = Agent(
    role="Web Researcher",
    goal="Find the most recent news about a given topic",
    backstory="An AI research assistant with expertise in real-time web searches and news analysis.",
    tools=[search_tool],
    verbose=True,
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Agent that writes a summary
writer_agent = Agent(
    
    role="News Analyst",
    
    # The agent's *goal* is the core objective it tries to achieve during its assigned tasks.
    # This helps the LLM stay focused on what matters most for this agent.
    goal="Write a clean, concise summary of the recent news",
    
    # The agent's *backstory* provides contextual depth to the agent.
    # This doesn't affect logic but significantly improves the LLM’s responses by grounding them in a persona.
    backstory="A clear communicator who distills complex news topics into easy-to-understand summaries.",
    verbose=True,
    
    # Whether or not the agent retains memory of prior interactions.
    # If set to True, the agent can use context from earlier in the run, helpful for coherent multi-step tasks.
    memory=True,
    llm=local_llm  # <-- use local LLM
)

# Task to perform the search
search_task = Task(
    description="Search the web for the latest news related to {topic}. Return key facts and headlines.",
    expected_output="A list of 5-10 relevant and recent headlines with 1-2 sentence summaries for each.",
    agent=search_agent
)

# Task to write the summary
summary_task = Task(
    description="Using the results from the search, write a well-structured summary report.",
    expected_output="A 3-paragraph summary of the most recent developments related to {topic}.",
    agent=writer_agent
)

# Create the crew
crew = Crew(
    agents=[search_agent, writer_agent],
    tasks=[search_task, summary_task],
    process=Process.sequential
)

# Run the crew
result = crew.kickoff(inputs={"topic": "AI regulation in Europe"})
print(result)
