# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 10:29:34 2025

@author: milos
"""
#%% Build a Question/Answering system over SQL data

"""
At a high-level, the steps are:

Convert question to SQL query: Model converts user input to a SQL query.
Execute SQL query: Execute the query.
Answer the question: Model responds to user input using the query results.
"""

#%%

from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

# pip install langchain langchain-community langgraph psycopg psycopg[binary]

#%% Connect to database

# Establish a connection to the PostgreSQL 'chinook' database using provided credentials.
db = SQLDatabase.from_uri("postgresql+psycopg://postgres:admin@localhost:5432/chinook")

# This helps confirm the type of database and SQL syntax that will be used for queries.
print(db.dialect)

# Retrieve and print a list of table names that are usable in the current database.
print(db.get_usable_table_names())

# Run a SQL query to select the first 10 rows from the 'Artist' table and print the results.
print(db.run("SELECT * FROM Artist LIMIT 10;"))

#%%

# Import Ollama model
model = ChatOllama(model='gpt-oss:120b-cloud')

toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect="SQLite",
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_message,
)


#%%

question = "Which country's customers spent the most?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

#%%

question = "Describe the playlisttrack table"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
    
#%%

question = "How many albums does alis in chain have?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
    
    
