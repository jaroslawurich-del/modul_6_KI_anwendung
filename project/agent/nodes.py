from llm.factory import ModelFactory
from rag.pipeline import search

llm = ModelFactory.llm()

def rag_node(state):

    result = search(state["question"])

    prompt = f"""
Nutze Kontext:

{result['context']}

Frage:
{state['question']}
"""

    res = llm.invoke(prompt)

    return {
        "answer": res.content,
        "sources": result["sources"]
    }