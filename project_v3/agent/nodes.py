from typing import TypedDict

from llm.factory import ModelFactory
from rag.pipeline import build_context


class AgentState(TypedDict):
    question: str
    answer: str


def rag_node(state: AgentState):

    question = state["question"]

    context = build_context(question)

    prompt = f"""
Du bist ein Dokumentenassistent.

Beantworte die Frage ausschließlich anhand des folgenden Kontextes.

Falls die Information nicht enthalten ist, antworte:

"Diese Information befindet sich nicht in den Dokumenten."

------------------------
Kontext:

{context}

------------------------

Frage:

{question}
"""

    llm = ModelFactory.chat()

    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content
    }