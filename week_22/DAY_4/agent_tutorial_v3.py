# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 10:58:21 2025

@author: milos
"""

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

# Setup LLM
llm = ChatOllama(model='gpt-oss:120b-cloud')

# Setup search tool
search = TavilySearch(
    k=3,
    tavily_api_key="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"
)

tools = [search]


# Create React agent
agent = create_agent(model=llm, 
                     tools=tools, 
                     system_prompt="You are a helpful assistant. Be concise and accurate.")

    
# Infinite loop to continuously accept user questions from the terminal
while True:
    # Prompt for user input (the question to ask the agent)
    query = input("\nYou: ")

    # If the input is "exit" or "quit" (case-insensitive), break the loop and end the session
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # The agent will use the tools and system prompt defined earlier
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

