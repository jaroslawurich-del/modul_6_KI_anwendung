from langgraph.graph import StateGraph
from langgraph.graph import END

from agent.nodes import (
    AgentState,
    rag_node,
)


workflow = StateGraph(AgentState)

workflow.add_node(
    "rag",
    rag_node,
)

workflow.set_entry_point("rag")

workflow.add_edge(
    "rag",
    END,
)

agent = workflow.compile()