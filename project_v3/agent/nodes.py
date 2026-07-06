from typing import TypedDict

from llm.factory import ModelFactory
from rag.pipeline import search


class AgentState(TypedDict):
    question: str
    answer: str


def rag_node(state: AgentState):

    question = state["question"]
    result = search(question)

    prompt = f"""
Nutze Kontext:

{result['context']}

Frage:
{question}
"""

    llm = ModelFactory.chat()

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content
    }