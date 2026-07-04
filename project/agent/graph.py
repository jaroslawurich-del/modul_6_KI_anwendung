from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import rag_node

graph = StateGraph(AgentState)

graph.add_node("rag", rag_node)
graph.set_entry_point("rag")
graph.add_edge("rag", END)

agent = graph.compile()